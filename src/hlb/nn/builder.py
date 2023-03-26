import torch
import torch.nn as nn
from ..config import hyp
from .speedyresnet import (
    BatchNorm,
    Conv,
    ConvGroup,
    FastGlobalMaxPooling,
    Linear,
    SpeedyResNet,
    TemperatureScaler,
)

from ..data import create_dataset

#############################################
#          Init Helper Functions            #
#############################################


def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
    c, (h, w) = x.shape[1], patch_shape
    return (
        x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)
    )  # TODO: Annotate?


def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n, c, h, w = patches.shape
    est_covariance = torch.cov(patches.view(n, c * h * w).t())
    eigenvalues, eigenvectors = torch.linalg.eigh(
        est_covariance, UPLO="U"
    )  # this is the same as saying we want our eigenvectors, with the specification that the matrix be an upper triangular matrix (instead of a lower-triangular matrix)
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(
        c * h * w, c, h, w
    ).flip(0)


# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
def init_whitening_conv(
    layer,
    train_set=None,
    num_examples=None,
    previous_block_data=None,
    pad_amount=None,
    freeze=True,
    whiten_splits=None,
):
    if train_set is not None and previous_block_data is None:
        if pad_amount > 0:
            previous_block_data = train_set[
                :num_examples, :, pad_amount:-pad_amount, pad_amount:-pad_amount
            ]  # if it's none, we're at the beginning of our network.
        else:
            previous_block_data = train_set[:num_examples, :, :, :]

    # chunking code to save memory for smaller-memory-size (generally consumer) GPUs
    if whiten_splits is None:
        previous_block_data_split = [
            previous_block_data
        ]  # If we're whitening in one go, then put it in a list for simplicity to reuse the logic below
    else:
        previous_block_data_split = previous_block_data.split(
            whiten_splits, dim=0
        )  # Otherwise, we split this into different chunks to keep things manageable

    eigenvalue_list, eigenvector_list = [], []
    for data_split in previous_block_data_split:
        eigenvalues, eigenvectors = get_whitening_parameters(
            get_patches(data_split, patch_shape=layer.weight.data.shape[2:])
        )
        eigenvalue_list.append(eigenvalues)
        eigenvector_list.append(eigenvectors)

    eigenvalues = torch.stack(eigenvalue_list, dim=0).mean(0)
    eigenvectors = torch.stack(eigenvector_list, dim=0).mean(0)
    # i believe the eigenvalues and eigenvectors come out in float32 for this because we implicitly cast it to float32 in the patches function (for numerical stability)
    set_whitening_conv(
        layer,
        eigenvalues.to(dtype=layer.weight.dtype),
        eigenvectors.to(dtype=layer.weight.dtype),
        freeze=freeze,
    )
    data = layer(previous_block_data.to(dtype=layer.weight.dtype))
    return data


def set_whitening_conv(conv_layer, eigenvalues, eigenvectors, eps=1e-2, freeze=True):
    shape = conv_layer.weight.data.shape
    conv_layer.weight.data[-eigenvectors.shape[0] :, :, :, :] = (
        eigenvectors / torch.sqrt(eigenvalues + eps)
    )[
        -shape[0] :, :, :, :
    ]  # set the first n filters of the weight data to the top n significant (sorted by importance) filters from the eigenvectors
    ## We don't want to train this, since this is implicitly whitening over the whole dataset
    ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    if freeze:
        conv_layer.weight.requires_grad = False


#############################################
#            Network Definition             #
#############################################

scaler = 2.0  ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    "init": round(
        scaler**-1 * hyp["net"]["base_depth"]
    ),  # 32  w/ scaler at base value
    "block1": round(
        scaler**0 * hyp["net"]["base_depth"]
    ),  # 64  w/ scaler at base value
    "block2": round(
        scaler**2 * hyp["net"]["base_depth"]
    ),  # 256 w/ scaler at base value
    "block3": round(
        scaler**3 * hyp["net"]["base_depth"]
    ),  # 512 w/ scaler at base value
    "num_classes": 10,
}


def make_net():
    data = create_dataset()
    # TODO: A way to make this cleaner??
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    whiten_conv_depth = 3 * hyp["net"]["whitening"]["kernel_size"] ** 2
    network_dict = nn.ModuleDict(
        {
            "initial_block": nn.ModuleDict(
                {
                    "whiten": Conv(
                        3,
                        whiten_conv_depth,
                        kernel_size=hyp["net"]["whitening"]["kernel_size"],
                        padding=0,
                    ),
                    "project": Conv(
                        whiten_conv_depth, depths["init"], kernel_size=1, norm=2.2
                    ),  # The norm argument means we renormalize the weights to be length 1 for this as the power for the norm, each step
                    "activation": nn.GELU(),
                }
            ),
            "residual1": ConvGroup(depths["init"], depths["block1"]),
            "residual2": ConvGroup(depths["block1"], depths["block2"]),
            "residual3": ConvGroup(depths["block2"], depths["block3"]),
            "pooling": FastGlobalMaxPooling(),
            "linear": Linear(
                depths["block3"], depths["num_classes"], bias=False, norm=5.0
            ),
            "temperature": TemperatureScaler(hyp["opt"]["scaling_factor"]),
        }
    )

    net = SpeedyResNet(network_dict)
    net = net.to(hyp["misc"]["device"])
    net = net.to(
        memory_format=torch.channels_last
    )  # to appropriately use tensor cores/avoid thrash while training
    net.train()
    net.half()  # Convert network to half before initializing the initial whitening layer.

    ## Initialize the whitening convolution
    with torch.no_grad():
        # Initialize the first layer to be fixed weights that whiten the expected input values of the network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
        init_whitening_conv(
            net.net_dict["initial_block"]["whiten"],
            data["train"]["images"].index_select(
                0,
                torch.randperm(
                    data["train"]["images"].shape[0],
                    device=data["train"]["images"].device,
                ),
            ),
            num_examples=hyp["net"]["whitening"]["num_examples"],
            pad_amount=hyp["net"]["pad_amount"],
            whiten_splits=5000,
        )  ## Hardcoded for now while we figure out the optimal whitening number
        ## If you're running out of memory (OOM) feel free to decrease this, but
        ## the index lookup in the dataloader may give you some trouble depending
        ## upon exactly how memory-limited you are

    return net
