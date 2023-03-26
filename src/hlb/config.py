import torch

from . import repo_basepath

# This is for testing that certain changes don't exceed some X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
# torch.cuda.set_per_process_memory_fraction(fraction=6.5/40., device=0) ## 40. GB is the maximum memory of the base A100 GPU

# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {"kernel_size": 3, "padding": "same", "bias": False}

batchsize = 1024
bias_scaler = 48
# To replicate the ~95.80%-accuracy-in-120-seconds runs, you can change the base_depth from 64->128, num_epochs from 10->90, ['ema'] epochs 9->78, and cutmix_size 0->9
hyp = {
    "opt": {
        "bias_lr": 1.64
        * bias_scaler
        / 512,  # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        "non_bias_lr": 1.64 / 512,
        "bias_decay": 1.05 * 6.45e-4 * batchsize / bias_scaler,
        "non_bias_decay": 1.05 * 6.45e-4 * batchsize,
        "scaling_factor": 1.0 / 9,
        "percent_start": 0.23,
        "loss_scale_scaler": 1.0
        / 128,  # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    "net": {
        "whitening": {
            "kernel_size": 2,
            "num_examples": 50000,
        },
        "batch_norm_momentum": 0.5,  # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        "cutmix_size": 3,
        "cutmix_epochs": 5,
        "pad_amount": 2,
        "base_depth": 64,  ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    "misc": {
        "ema": {
            "epochs": 10,  # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            "decay_base": 0.95,
            "decay_pow": 3,
            "every_n_steps": 5,
        },
        "train_epochs": 12.6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_location": repo_basepath / "artifacts/hlb/data.pt",
    },
}
