"""Layer wise frobenius norm error comparison of applying LPLR SVD on LlaMa

RUN Commands

export OUTPUT_DIRECTORY="./misc/llama/layer-wise-lplr-naive-quant-comparison"
export MODEL_DIRECTORY="./artifacts/LLAMA/llama-7b-hf/"
export LOGURU_LEVEL=INFO
stdbuf -oL python scripts/llama/layer_wise_lplr_svd_quantization.py --model-directory $MODEL_DIRECTORY --output-directory $OUTPUT_DIRECTORY --b1 8 --b2 8 --b_nq 8 --cr 1 --map-location "cuda:1" 2>&1 | stdbuf -oL tee -i $OUTPUT_DIRECTORY/lplr-svd-quantization-$(date +%m%d%H%M%S).log
"""

import json
import pathlib
import sys
from math import ceil

import fire
import torch
from loguru import logger

from lplr.compressors import lplr_svd
from lplr.quantizers import quantize
from lplr.utils import maximum_output_rank
from shared.error import relative_tensor_error
from natsort import natsorted, ns


def main(
    model_directory: pathlib.Path | str,
    output_directory: pathlib.Path | str,
    b1: int = 8,
    b2: int = 8,
    cr: float = 1,
    b_nq: int = 8,
    map_location: str = "cuda:0",
    sketch: str = "Gaussian",
    **kwargs,
):
    """_summary_

    Args:
        model_directory (pathlib.Path | str): _description_
        output_directory (pathlib.Path | str): _description_
        b1 (int, optional): _description_. Defaults to 8.
        b2 (int, optional): _description_. Defaults to 8.
        b_nq (int, optional): Bit budget for naive quantizer. Defaults to 8.
        cr (float, optional): 0 <= cr <= 1, Defaults to 1. CR = New Model Size / Original Model Size, for compressing a model, must be less than 1.
        map_location (_type_, optional): _description_. Defaults to "cuda:0".
    """
    logger.info(f"Input Arguments: {sys.argv}")
    model_directory = pathlib.Path(model_directory)
    model_files = natsorted(model_directory.glob("*.bin"), alg=ns.IC)

    output_directory = pathlib.Path(output_directory)
    logger.info(f"Applying LPLR on error matrices")
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

                col_input = weights
                row_input = weights.T

                col_output_rank = maximum_output_rank(cr, b1, b2, b_nq, col_input.shape)
                row_output_rank = maximum_output_rank(cr, b1, b2, b_nq, row_input.shape)

                logger.debug(
                    f"Name: {name} "
                    f"Shape: {weights.shape} "
                    f"Row Output Rank: {row_output_rank} "
                    f"Col Output Rank: {col_output_rank}"
                )

                if b1 == b2:
                    assert col_output_rank == row_output_rank

                # No normalize and shift, as α ~ 1 and β ~ 0
                col_sketch = lplr_svd(
                    col_input, col_output_rank, b1, b2, sketch=sketch, **kwargs
                )
                row_sketch = lplr_svd(
                    row_input, row_output_rank, b1, b2, sketch=sketch, **kwargs
                )

                col_sketch_err = relative_tensor_error(col_input, col_sketch)
                row_sketch_err = relative_tensor_error(row_input, row_sketch)

                naive_quant = quantize(weights, b_nq, preserve_original_dtype=True)
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
                logger.info(f"Name: {name} " f"Shape: {weights.shape} " f"Unmodified")


if __name__ == "__main__":
    fire.Fire(main)
