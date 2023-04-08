#!/usr/bin/env python
# coding: utf-8

# This notebook loads a pre-trained HyperLightspeedBench (HLB) neural network for image classification on CIFAR-10 dataset and experiments with low-precision low-rank approximation of the weight matrices of the neural network.


import pathlib
from functools import partial
from typing import Callable, Final

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from hlb import repo_basepath
from hlb.config import hyp
from hlb.nn.speedyresnet import (
    BatchNorm,
    Conv,
    ConvGroup,
    FastGlobalMaxPooling,
    Linear,
    SpeedyResNet,
    TemperatureScaler,
)
from hlb.utils import get_batches
from lplr.compressors import direct_svd_quant, lplr
from lplr.quantizers import quantize

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def evaluate_models(model: nn.Module, benchmark_model):
    model.eval()
    benchmark_model.eval()
    model = model.to(device).float()
    benchmark_model = benchmark_model.to(device).float()
    data = torch.load(hyp["misc"]["data_location"], map_location=device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction="none")

    eval_batchsize = 2500
    from collections import defaultdict

    loss_list_val, acc_list = defaultdict(list), defaultdict(list)

    model_types = ("Test model", "Benchmark Model")
    with torch.no_grad():
        for inputs, targets in get_batches(data, key="eval", batchsize=eval_batchsize):
            input_tensors = inputs.float()

            for mm, mt in zip((model, benchmark_model), model_types):
                outputs = mm(input_tensors)
                loss_val = loss_fn(outputs, targets).float().mean()
                acc_val = (outputs.argmax(-1) == targets.argmax(-1)).float().mean()
                logger.trace(f"loss {loss_val:.3f} acc {acc_val:.3f} for {mt}")

                loss_list_val[mt].append(loss_val)
                acc_list[mt].append(acc_val)

    for mt in model_types:
        avg_val_acc = torch.mean(torch.tensor(acc_list[mt])).item()
        avg_val_loss = torch.mean(torch.tensor(loss_list_val[mt])).item()

        logger.debug(
            f"Avg Validation Accuracy: {avg_val_acc:.2f} and Avg Validation Loss {avg_val_loss} for {mt}"
        )
    return (
        torch.mean(torch.tensor(acc_list["Test model"])).item(),
        torch.mean(torch.tensor(loss_list_val["Test model"])).item(),
    )


def quantize_layers(
    model: nn.Module, compressor: Callable[[np.ndarray], np.ndarray] = lplr
) -> nn.Module:
    from copy import deepcopy

    output_model = deepcopy(model)
    # b1 = 8
    # b2 = 8
    # frac = 0.9
    for name, param in output_model.named_parameters():
        model_param = param.to(device).detach()
        param_shape = model_param.shape
        logger.trace(f"Applying LPLR on {name} with shape {param_shape}")
        if param.ndim >= 2:
            reshaped_param = model_param.reshape(param_shape[0], param_shape[1], -1)
            out_param = torch.zeros_like(reshaped_param)
            for idxs in range(reshaped_param.shape[-1]):
                out_param[:, :, idxs] = compressor(reshaped_param[:, :, idxs])
            param.data = out_param.reshape(param_shape)
        elif param.ndim == 1:
            param.data = quantize(model_param)
    return output_model


model_location = repo_basepath / "artifacts/hlb/checkpoints/trained-speedyresnet.pt"
model = torch.load(model_location, map_location=device)


def comp(param, fraction, b1, b2):
    from math import ceil

    rank = min(param.shape)
    try:
        out_rank = ceil(fraction * rank)
        assert isinstance(out_rank, int)
    except AssertionError as ae:
        logger.error(f"Wrong Out rank: {out_rank}")
        raise ae
    logger.debug(f"Shape: {param.shape} Input Rank: {rank} Output Rank: {out_rank}")
    return lplr(param, out_rank, b1, b2)


def comp_direct_svd(param, fraction, b1, b2):
    from math import ceil

    rank = min(param.shape)
    try:
        out_rank = ceil(fraction * rank)
        assert isinstance(out_rank, int)
    except AssertionError as ae:
        logger.error(f"Wrong Out rank: {out_rank}")
        raise ae
    logger.debug(f"Shape: {param.shape} Input Rank: {rank} Output Rank: {out_rank}")
    return direct_svd_quant(param, out_rank, b1, b2)


def plot_accs(keys, acc, acc_direct_svd, rvar, b1, b2, outfile):
    # Relies on Cpython ordering
    plt.plot(keys, acc, marker="o", color="blue", markersize=10, label="LPLR")
    plt.fill_between(keys, acc - rvar, acc + rvar)
    plt.plot(
        keys,
        acc_direct_svd,
        marker="x",
        color="red",
        markersize=10,
        label="Direct SVD Quant",
    )
    plt.xlabel("Fraction of singular values retained")
    plt.xticks(keys)
    plt.xticks(rotation=90)
    plt.ylabel("Test accuracy")
    plt.title(f"HLB accuracy on CIFAR-10 with B1 = {b1} and B2 = {b2}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    # plt.show()


def main(
    b1=8, b2=8, fracs: tuple[str] = ("0.7", "0.75", "0.8", "0.9", "0.95", "1.0", "1.1")
):
    from collections import defaultdict

    # Bit budgets
    B1 = b1
    B2 = b2

    # Quantize models using direct SVD quant
    quantized_models_direct_svd = {
        f: quantize_layers(
            model,
            compressor=partial(comp_direct_svd, fraction=float(f), b1=B1, b2=B2),
        )
        for f in fracs
    }
    # Evaluate direct-SVD quantized models
    acc_direct_svd = {}
    loss_direct_svd = {}
    for f, qm in quantized_models_direct_svd.items():
        (acc_direct_svd[f], loss_direct_svd[f]) = evaluate_models(qm, model)
        logger.info(
            f"Direct-SVD: Computing fraction {f} with accuracy {acc_direct_svd[f]:.3f} and loss {loss_direct_svd[f]:.3f}"
        )

    random_accs = []
    # Quantize models using LPLR
    for seed in range(5):
        global SEED
        SEED = seed
        logger.info(f"Using seed = {SEED}")
        quantized_models = {
            f: quantize_layers(
                model, compressor=partial(comp, fraction=float(f), b1=B1, b2=B2)
            )
            for f in fracs
        }

        # Evaluate LPLR quantized models
        acc = {}
        loss = {}

        for f, qm in quantized_models.items():
            (acc[f], loss[f]) = evaluate_models(qm, model)
            logger.info(
                f"Computing fraction {f} with accuracy {acc[f]:.3f} and loss {loss[f]:.3f}"
            )

        random_accs.append(list(acc.values()))
    random_accs = np.array(random_accs)

    plot_accs(
        fracs,
        random_accs.mean(axis=0),
        acc_direct_svd.values(),
        3 * random_accs.std(axis=0),
        B1,
        B2,
        pathlib.Path("misc") / "hlb" / f"seed_averaged_{B1}_{B2}.png",
    )


if __name__ == "__main__":
    fire.Fire(main)
