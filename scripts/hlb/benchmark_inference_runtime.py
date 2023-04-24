import pathlib
from functools import partial
from typing import Callable, Final
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from shared import repo_basepath
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
from lplr.compressors import lplr
from lplr.quantizers import quantize

from time import perf_counter_ns
from sys import argv


def main(precision: str = "half"):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model_location: pathlib.Path = (
        repo_basepath / "artifacts/hlb/checkpoints/trained-speedyresnet.pt"
    )
    model: nn.Module = torch.load(model_location, map_location=device)
    data: torch.Tensor = torch.load(hyp["misc"]["data_location"], map_location=device)[
        "eval"
    ]["images"].to(memory_format=torch.channels_last)

    logger.info(data.shape)

    match precision:
        case "half":
            model = model.half()
            data = data.half()
        case "float":
            model = model.float()
            data = data.float()
        case "double":
            model = model.double()
            data = data.double()
        case _:
            logger.error(f"{precision} not supported")
            raise ValueError(f"{precision} not supported")

    batch_size = 1500
    logger.info(f"Using batchsize = {batch_size} for evaluation")
    indexed_data = data[:batch_size, ...]

    model.eval()

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()  ## clean up any pre-net setup operations
    starter.record()
    _ = model(indexed_data)
    ender.record()
    torch.cuda.synchronize()

    total_time_seconds = 1e-3 * starter.elapsed_time(ender)

    logger.info(f"{total_time_seconds} s elapsed")


if __name__ == "__main__":
    main(argv[1])
