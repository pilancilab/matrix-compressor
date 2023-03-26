# Note: The one change we need to make if we're in Colab is to uncomment this below block.
# If we are in an ipython session or a notebook, clear the state to avoid bugs
"""
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""
import copy
import math
import os
import pathlib
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

from .data.builder import create_dataset

from .config import batchsize, bias_scaler, default_conv_kwargs, hyp
from .nn.speedyresnet import (
    Conv,
    ConvGroup,
    FastGlobalMaxPooling,
    Linear,
    SpeedyResNet,
    TemperatureScaler,
)
from .utils import get_batches, init_split_parameter_dictionaries
from .nn import NetworkEMA, make_net

## <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
# torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). That said, if there's
# ways this code could be improved and cleaned up, please do open a PR on the GitHub repo. Your support and help is much appreciated for this
# project! :)


## Hey look, it's the soft-targets/label-smoothed loss! Native to PyTorch. Now, _that_ is pretty cool, and simplifies things a lot, to boot! :D :)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction="none")

logging_columns_list = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "ema_val_acc",
    "total_time_seconds",
]


# define the printing function and print the column heads
def print_training_details(
    columns_list,
    separator_left="|  ",
    separator_right="  ",
    final="|",
    column_heads_only=False,
    is_final_entry=False,
):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print("-" * (len(print_string)))  # print the top bar
        print(print_string)
        print("-" * (len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print("-" * (len(print_string)))  # print the final output bar


print_training_details(
    logging_columns_list, column_heads_only=True
)  ## print out the training column heads before we print the actual content for each run.

########################################
#           Train and Eval             #
########################################


def main():
    data = create_dataset()
    # Initializing constants for the whole run.
    net_ema = None  ## Reset any existing network emas, we want to have _something_ to check for existence so we can initialize the EMA right from where the network is during training
    ## (as opposed to initializing the network_ema from the randomly-initialized starter network, then forcing it to play catch-up all of a sudden in the last several epochs)

    total_time_seconds = 0.0
    current_steps = 0.0

    # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
    num_steps_per_epoch = len(data["train"]["images"]) // batchsize
    total_train_steps = math.ceil(num_steps_per_epoch * hyp["misc"]["train_epochs"])
    ema_epoch_start = (
        math.floor(hyp["misc"]["train_epochs"]) - hyp["misc"]["ema"]["epochs"]
    )

    ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
    ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
    ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
    projected_ema_decay_val = (
        hyp["misc"]["ema"]["decay_base"] ** hyp["misc"]["ema"]["every_n_steps"]
    )

    # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
    pct_start = hyp["opt"][
        "percent_start"
    ]  # * (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))

    # Get network
    net = make_net()

    ## Stowing the creation of these into a helper function to make things a bit more readable....
    non_bias_params, bias_params = init_split_parameter_dictionaries(net)

    # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
    opt = torch.optim.SGD(**non_bias_params)
    opt_bias = torch.optim.SGD(**bias_params)

    ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
    ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
    initial_div_factor = 1e16  # basically to make the initial lr ~0 or so :D
    final_lr_ratio = 0.05  # Actually pretty important, apparently!
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=non_bias_params["lr"],
        pct_start=pct_start,
        div_factor=initial_div_factor,
        final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
        total_steps=total_train_steps,
        anneal_strategy="linear",
        cycle_momentum=False,
    )
    lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(
        opt_bias,
        max_lr=bias_params["lr"],
        pct_start=pct_start,
        div_factor=initial_div_factor,
        final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
        total_steps=total_train_steps,
        anneal_strategy="linear",
        cycle_momentum=False,
    )

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()  ## clean up any pre-net setup operations
    checkpoint_folder = pathlib.Path(
        f"checkpoints/run-{int(datetime.now().timestamp())}",
    )
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    if (
        True
    ):  ## Sometimes we need a conditional/for loop here, this is placed to save the trouble of needing to indent
        for epoch in range(math.ceil(hyp["misc"]["train_epochs"])):
            #################
            # Training Mode #
            #################
            torch.cuda.synchronize()
            starter.record()
            net.train()

            cutmix_size = (
                hyp["net"]["cutmix_size"]
                if epoch >= hyp["misc"]["train_epochs"] - hyp["net"]["cutmix_epochs"]
                else 0
            )
            epoch_fraction = (
                1
                if epoch + 1 < hyp["misc"]["train_epochs"]
                else hyp["misc"]["train_epochs"] % 1
            )  # We need to know if we're running a partial epoch or not.

            for epoch_step, (inputs, targets) in enumerate(
                get_batches(
                    data,
                    key="train",
                    batchsize=batchsize,
                    epoch_fraction=epoch_fraction,
                    cutmix_size=cutmix_size,
                )
            ):
                ## Run everything through the network
                outputs = net(inputs)

                loss_batchsize_scaler = (
                    512 / batchsize
                )  # to scale to keep things at a relatively similar amount of regularization when we change our batchsize since we're summing over the whole batch
                ## If you want to add other losses or hack around with the loss, you can do that here.
                loss = (
                    loss_fn(outputs, targets)
                    .mul(hyp["opt"]["loss_scale_scaler"] * loss_batchsize_scaler)
                    .sum()
                    .div(hyp["opt"]["loss_scale_scaler"])
                )  ## Note, as noted in the original blog posts, the summing here does a kind of loss scaling
                ## (and is thus batchsize dependent as a result). This can be somewhat good or bad, depending...

                # we only take the last-saved accs and losses from train
                if epoch_step % 50 == 0:
                    train_acc = (
                        (outputs.detach().argmax(-1) == targets.argmax(-1))
                        .float()
                        .mean()
                        .item()
                    )
                    train_loss = loss.detach().cpu().item() / (
                        batchsize * loss_batchsize_scaler
                    )

                loss.backward()

                ## Step for each optimizer, in turn.
                opt.step()
                opt_bias.step()

                # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
                lr_sched.step()
                lr_sched_bias.step()

                ## Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
                opt.zero_grad(set_to_none=True)
                opt_bias.zero_grad(set_to_none=True)
                current_steps += 1

                if (
                    epoch >= ema_epoch_start
                    and current_steps % hyp["misc"]["ema"]["every_n_steps"] == 0
                ):
                    ## Initialize the ema from the network at this point in time if it does not already exist.... :D
                    if net_ema is None:  # don't snapshot the network yet if so!
                        net_ema = NetworkEMA(net)
                        continue
                    # We warm up our ema's decay/momentum value over training exponentially according to the hyp config dictionary (this lets us move fast, then average strongly at the end).
                    net_ema.update(
                        net,
                        decay=projected_ema_decay_val
                        * (current_steps / total_train_steps)
                        ** hyp["misc"]["ema"]["decay_pow"],
                    )

            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)

            ####################
            # Evaluation  Mode #
            ####################
            net.eval()

            eval_batchsize = 2500
            assert (
                data["eval"]["images"].shape[0] % eval_batchsize == 0
            ), "Error: The eval batchsize must evenly divide the eval dataset (for now, we don't have drop_remainder implemented yet)."
            loss_list_val, acc_list, acc_list_ema = [], [], []

            with torch.no_grad():
                for inputs, targets in get_batches(
                    data, key="eval", batchsize=eval_batchsize
                ):
                    if epoch >= ema_epoch_start:
                        outputs = net_ema(inputs)
                        acc_list_ema.append(
                            (outputs.argmax(-1) == targets.argmax(-1)).float().mean()
                        )
                    outputs = net(inputs)
                    loss_list_val.append(loss_fn(outputs, targets).float().mean())
                    acc_list.append(
                        (outputs.argmax(-1) == targets.argmax(-1)).float().mean()
                    )

                val_acc = torch.stack(acc_list).mean().item()
                ema_val_acc = None
                # TODO: We can fuse these two operations (just above and below) all-together like :D :))))
                if epoch >= ema_epoch_start:
                    ema_val_acc = torch.stack(acc_list_ema).mean().item()

                val_loss = torch.stack(loss_list_val).mean().item()

            output_path = checkpoint_folder / f"init-{epoch}"
            torch.save(net, output_path)

            # We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
            ## Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
            ## more heinous than this, unfortunately. So we switched to the "more simple" version of this!
            format_for_table = (
                lambda x, locals: (f"{locals[x]}".rjust(len(x)))
                if type(locals[x]) == int
                else "{:0.4f}".format(locals[x]).rjust(len(x))
                if locals[x] is not None
                else " " * len(x)
            )

            # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
            ## We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
            print_training_details(
                list(
                    map(
                        partial(format_for_table, locals=locals()), logging_columns_list
                    )
                ),
                is_final_entry=(epoch >= math.ceil(hyp["misc"]["train_epochs"] - 1)),
            )
    return ema_val_acc  # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)


if __name__ == "__main__":
    acc_list = []
    for run_num in range(25):
        acc_list.append(torch.tensor(main()))
    print(
        "Mean and variance:",
        (
            torch.mean(torch.stack(acc_list)).item(),
            torch.var(torch.stack(acc_list)).item(),
        ),
    )
