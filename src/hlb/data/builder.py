import os
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from ..config import hyp
from ..utils import batch_normalize_images
from .. import repo_basepath

def create_dataset():
    #############################################
    #                Dataloader                 #
    #############################################

    if not os.path.exists(hyp["misc"]["data_location"]):
        transform = transforms.Compose([transforms.ToTensor()])

        dataset_path = repo_basepath / "artifacts/hlb/cifar10/"
        cifar10 = torchvision.datasets.CIFAR10(
            dataset_path, download=True, train=True, transform=transform
        )
        cifar10_eval = torchvision.datasets.CIFAR10(
            dataset_path, download=False, train=False, transform=transform
        )

        # use the dataloader to get a single batch of all of the dataset items at once.
        train_dataset_gpu_loader = torch.utils.data.DataLoader(
            cifar10,
            batch_size=len(cifar10),
            drop_last=True,
            shuffle=True,
            num_workers=2,
            persistent_workers=False,
        )
        eval_dataset_gpu_loader = torch.utils.data.DataLoader(
            cifar10_eval,
            batch_size=len(cifar10_eval),
            drop_last=True,
            shuffle=False,
            num_workers=1,
            persistent_workers=False,
        )

        train_dataset_gpu = {}
        eval_dataset_gpu = {}

        train_dataset_gpu["images"], train_dataset_gpu["targets"] = [
            item.to(device=hyp["misc"]["device"], non_blocking=True)
            for item in next(iter(train_dataset_gpu_loader))
        ]
        eval_dataset_gpu["images"], eval_dataset_gpu["targets"] = [
            item.to(device=hyp["misc"]["device"], non_blocking=True)
            for item in next(iter(eval_dataset_gpu_loader))
        ]

        cifar10_std, cifar10_mean = torch.std_mean(
            train_dataset_gpu["images"], dim=(0, 2, 3)
        )  # dynamically calculate the std and mean from the data. this shortens the code and should help us adapt to new datasets!

        # preload with our mean and std
        batch_norm_im_partial = partial(
            batch_normalize_images, mean=cifar10_mean, std=cifar10_std
        )

        ## Batch normalize datasets, now. Wowie. We did it! We should take a break and make some tea now.
        train_dataset_gpu["images"] = batch_norm_im_partial(
            train_dataset_gpu["images"]
        )
        eval_dataset_gpu["images"] = batch_norm_im_partial(eval_dataset_gpu["images"])

        data = {"train": train_dataset_gpu, "eval": eval_dataset_gpu}

        ## Convert dataset to FP16 now for the rest of the process....
        data["train"]["images"] = data["train"]["images"].half().requires_grad_(False)
        data["eval"]["images"] = data["eval"]["images"].half().requires_grad_(False)

        # Convert this to one-hot to support the usage of cutmix (or whatever strange label tricks/magic you desire!)
        data["train"]["targets"] = F.one_hot(data["train"]["targets"]).half()
        data["eval"]["targets"] = F.one_hot(data["eval"]["targets"]).half()

        torch.save(data, hyp["misc"]["data_location"])

    else:
        ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
        ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
        ## hyp dictionary, then we should be good. :)
        data = torch.load(hyp["misc"]["data_location"])

    ## As you'll note above and below, one difference is that we don't count loading the raw data to GPU since it's such a variable operation, and can sort of get in the way
    ## of measuring other things. That said, measuring the preprocessing (outside of the padding) is still important to us.

    # Pad the GPU training dataset
    if hyp["net"]["pad_amount"] > 0:
        ## Uncomfortable shorthand, but basically we pad evenly on all _4_ sides with the pad_amount specified in the original dictionary
        data["train"]["images"] = F.pad(
            data["train"]["images"].float() if hyp["misc"]["device"]=="cpu" else data["train"]["images"], (hyp["net"]["pad_amount"],) * 4, "reflect",
        )

    return data
