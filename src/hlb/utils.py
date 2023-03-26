import torch
from .config import hyp


def batch_normalize_images(input_images, mean, std):
    return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


#############################################
#            Data Preprocessing             #
#############################################


## This is actually (I believe) a pretty clean implementation of how to do something like this, since shifted-square masks unique to each depth-channel can actually be rather
## tricky in practice. That said, if there's a better way, please do feel free to submit it! This can be one of the harder parts of the code to understand (though I personally get
## stuck on the fold/unfold process for the lower-level convolution calculations.
def make_random_square_masks(inputs, mask_size):
    ##### TODO: Double check that this properly covers the whole range of values. :'( :')
    if mask_size == 0:
        return None  # no need to cutout or do anything like that since the patch_size is set to 0
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    # seed centers of squares to cutout boxes from, in one dimension each
    mask_center_y = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even)
    mask_center_x = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even)

    # measure distance, using the center as a reference point
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(
        1, 1, in_shape[-2], 1
    ) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(
        1, 1, 1, in_shape[-1]
    ) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_y_dists <= mask_size // 2
    )
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_x_dists <= mask_size // 2
    )

    final_mask = (
        to_mask_y * to_mask_x
    )  ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D

    return final_mask


def batch_cutmix(inputs, targets, patch_size):
    with torch.no_grad():
        batch_permuted = torch.randperm(inputs.shape[0], device=inputs.device)
        cutmix_batch_mask = make_random_square_masks(inputs, patch_size)
        if cutmix_batch_mask is None:
            return (
                inputs,
                targets,
            )  # if the mask is None, then that's because the patch size was set to 0 and we will not be using cutmix today.
        # We draw other samples from inside of the same batch
        cutmix_batch = torch.where(
            cutmix_batch_mask, torch.index_select(inputs, 0, batch_permuted), inputs
        )
        cutmix_targets = torch.index_select(targets, 0, batch_permuted)
        # Get the percentage of each target to mix for the labels by the % proportion of pixels in the mix
        portion_mixed = float(patch_size**2) / (inputs.shape[-2] * inputs.shape[-1])
        cutmix_labels = portion_mixed * cutmix_targets + (1.0 - portion_mixed) * targets
        return cutmix_batch, cutmix_labels


def batch_crop(inputs, crop_size):
    with torch.no_grad():
        crop_mask_batch = make_random_square_masks(inputs, crop_size)
        cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(
            inputs.shape[0], inputs.shape[1], crop_size, crop_size
        )
        return cropped_batch


def batch_flip_lr(batch_images, flip_chance=0.5):
    with torch.no_grad():
        # TODO: Is there a more elegant way to do this? :') :'((((
        return torch.where(
            torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance,
            torch.flip(batch_images, (-1,)),
            batch_images,
        )


# TODO: Could we jit this in the (more distant) future? :)
@torch.no_grad()
def get_batches(data_dict, key, batchsize, epoch_fraction=1.0, cutmix_size=None):
    num_epoch_examples = len(data_dict[key]["images"])
    shuffled = torch.randperm(
        num_epoch_examples, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    if epoch_fraction < 1:
        shuffled = shuffled[
            : batchsize * round(epoch_fraction * shuffled.shape[0] / batchsize)
        ]  # TODO: Might be slightly inaccurate, let's fix this later... :) :D :confetti: :fireworks:
        num_epoch_examples = shuffled.shape[0]
    crop_size = 32
    ## Here, we prep the dataset by applying all data augmentations in batches ahead of time before each epoch, then we return an iterator below
    ## that iterates in chunks over with a random derangement (i.e. shuffled indices) of the individual examples. So we get perfectly-shuffled
    ## batches (which skip the last batch if it's not a full batch), but everything seems to be (and hopefully is! :D) properly shuffled. :)
    if key == "train":
        images = batch_crop(
            data_dict[key]["images"], crop_size
        )  # TODO: hardcoded image size for now?
        images = batch_flip_lr(images)
        images, targets = batch_cutmix(
            images, data_dict[key]["targets"], patch_size=cutmix_size
        )
    else:
        images = data_dict[key]["images"]
        targets = data_dict[key]["targets"]

    # Send the images to an (in beta) channels_last to help improve tensor core occupancy (and reduce NCHW <-> NHWC thrash) during training
    images = images.to(memory_format=torch.channels_last)
    for idx in range(num_epoch_examples // batchsize):
        if (
            not (idx + 1) * batchsize > num_epoch_examples
        ):  ## Use the shuffled randperm to assemble individual items into a minibatch
            yield images.index_select(
                0, shuffled[idx * batchsize : (idx + 1) * batchsize]
            ), targets.index_select(
                0, shuffled[idx * batchsize : (idx + 1) * batchsize]
            )  ## Each item is only used/accessed by the network once per epoch. :D


def init_split_parameter_dictionaries(network):
    params_non_bias = {
        "params": [],
        "lr": hyp["opt"]["non_bias_lr"],
        "momentum": 0.85,
        "nesterov": True,
        "weight_decay": hyp["opt"]["non_bias_decay"],
        "foreach": True,
    }
    params_bias = {
        "params": [],
        "lr": hyp["opt"]["bias_lr"],
        "momentum": 0.85,
        "nesterov": True,
        "weight_decay": hyp["opt"]["bias_decay"],
        "foreach": True,
    }

    for name, p in network.named_parameters():
        if p.requires_grad:
            if "bias" in name:
                params_bias["params"].append(p)
            else:
                params_non_bias["params"].append(p)
    return params_non_bias, params_bias
