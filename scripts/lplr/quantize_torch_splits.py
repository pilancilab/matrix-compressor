import sys
from math import ceil
from pathlib import Path

import fire
import torch
from loguru import logger

from lplr.compressors import lplr
from shared import relative_tensor_error


def quantize_weight_dict(
    weight_dict: dict[str, torch.Tensor], b1: int, b2: int, f: float
):
    out_layers = {}
    for name, param in weight_dict.items():
        shape = param.shape
        assumed_rank = ceil(f * min(shape))
        if param.ndim == 2:
            q_param = lplr(param, r=assumed_rank, B1=b1, B2=b2)
            logger.debug(
                f"Quantized layer: {name} with shape {shape} using B1={b1} B2={b2} "
                f"to rank={assumed_rank} and rel error={relative_tensor_error(param, q_param)}",
            )

        else:
            q_param = param
            logger.debug(f"Layer {name} with shape {shape} unmodified")
        out_layers[name] = q_param.to("cpu").clone()
    return out_layers


def main(
    in_path: Path,
    out_path: Path,
    b1: int = 16,
    b2: int = 16,
    rank_fraction: float = 0.7,
    map_device: str = "cpu",
):
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        logger.error(f"{in_path} input path not found, terminating!")
        sys.exit(-1)

    if in_path.is_dir() and (out_path.exists() and not out_path.is_dir()):
        logger.error(
            f"{out_path} must be a directory when {in_path} is a directory, terminating!"
        )
        sys.exit(-2)

    if not out_path.exists() and in_path.is_dir():
        logger.warning(f"{out_path} does not exist, creating directory")
        out_path.mkdir(exist_ok=True, parents=True)

    if in_path.is_dir():
        # If output path is not a directory, it would have errored out already
        for model_file in in_path.glob("*.bin"):
            logger.debug(f"Loading file {model_file}")
            out_name = model_file.name
            q_dict = quantize_weight_dict(
                torch.load(model_file, map_location=map_device), b1, b2, rank_fraction
            )
            logger.debug(f"Writing to file {out_path / out_name}")
            torch.save(q_dict, out_path / out_name)
    elif in_path.is_file():
        logger.debug(f"Loading file {in_path}")
        q_dict = quantize_weight_dict(
            torch.load(in_path, map_location=map_device), b1, b2, rank_fraction
        )
        if out_path.is_dir():
            out_name = in_path.name
            logger.debug(f"Writing to file {out_path / out_name}")
            torch.save(q_dict, out_path / out_name)
        else:
            logger.debug(f"Writing to file {out_path}")
            torch.save(q_dict, out_path)
    else:
        logger.error(
            f"Unknown file like object passed, {in_path}, cannot write to this location"
        )
        sys.exit(-3)


if __name__ == "__main__":
    fire.Fire(main)
