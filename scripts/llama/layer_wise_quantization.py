import json
import pathlib
import sys
from math import ceil

import fire
import torch
from loguru import logger

from lplr.compressors import lplr
from lplr.quantizers import quantize
from shared.error import relative_tensor_error
from natsort import natsorted, ns

def main(
    model_directory: pathlib.Path | str,
    output_directory: pathlib.Path | str,
    b1: int = 8,
    b2: int = 8,
    cr: float = 1,
    map_location: str = "cuda:0",
):
    """_summary_

    Args:
        model_directory (pathlib.Path | str): _description_
        output_directory (pathlib.Path | str): _description_
        b1 (int, optional): _description_. Defaults to 8.
        b2 (int, optional): _description_. Defaults to 8.
        cr (float, optional): 0 <= cr <= 1, Defaults to 1. CR = New Model Size / Original Model Size, for compressing a model, must be less than 1.
        map_location (_type_, optional): _description_. Defaults to "cuda:0".
    """
    logger.info(f"Input Arguments: {sys.argv}")
    model_directory = pathlib.Path(model_directory)
    model_files = natsorted(model_directory.glob("*.bin"), alg=ns.IC)

    output_directory = pathlib.Path(output_directory)

    for f in model_files:
        layer_set = torch.load(f, map_location=map_location)

        for name, weights in layer_set.items():
            transpose = False
            original_dtype = weights.dtype
            if weights.ndim == 2:
                if weights.shape[1] > weights.shape[0]:
                    weights = weights.T
                    transpose = True
                min_dim = weights.shape[1]
                max_dim = weights.shape[0]
                assert max_dim >= min_dim

                output_rank = ceil(cr * ((min_dim * max_dim) / (min_dim + max_dim)))

                row_sketch = lplr(weights.T, output_rank, b1, b2)
                col_sketch = lplr(weights, output_rank, b1, b2)

                row_sketch_err = relative_tensor_error(weights.T, row_sketch)
                col_sketch_err = relative_tensor_error(weights, col_sketch)

                naive_quant = quantize(weights, b1, preserve_original_dtype=True)
                naive_quant_err = relative_tensor_error(weights, naive_quant)
                
                if transpose:
                    row_sketch_err, col_sketch_err = col_sketch_err, row_sketch_err
                logger.info(
                    f"Name: {name} "
                    f"Shape: {weights.shape} "
                    f"Row Sketch Error: {row_sketch_err:.3f} "
                    f"Col Sketch Error: {col_sketch_err:.3f} "
                    f"Dtype: {original_dtype} "
                    f"Naive Quant Err: {naive_quant_err:.3f} "
                    f"Compression Ratio: {cr}"
                )
            else:
                logger.info(
                    f"Name: {name} "
                    f"Shape: {weights.shape} "
                    f"Unmodified"
                )


if __name__ == "__main__":
    fire.Fire(main)
