"""Usage

python scripts/mnetv3/save_quantized_splits.py --input-data-location artifacts/custom_data/knn/mnetv3/cifar10/cifar10-train-embeddings.pt --output-data-location artifacts/custom_data/knn/mnetv3/cifar10-quantized/train --map-location cpu --b1 8 --b2 8 --b-nq 2 --cr 1
"""
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime
from math import ceil, floor

import fire
import torch
from loguru import logger
from tqdm import tqdm

from lplr.compressors import direct_svd_quant, lplr
from lplr.quantizers import quantize
from lplr.utils import maximum_output_rank
from mnet.utils import extract_embeddings
from shared import repo_basepath
from shared.error import relative_tensor_error


def main(
    output_data_location,
    input_data_location,
    b1: int = 8,
    b2: int = 8,
    cr: float = 1,
    b_nq: int = 8,
    map_location: str = "cuda:0",
    sketch: str = "Gaussian",
    force=False,
    logfile = None,
    **kwargs,
):
    try:
        temp_log_file = tempfile.NamedTemporaryFile(mode="w+", delete=True)
        logger.add(sink = temp_log_file, level = "TRACE", colorize=False)
        output_dir = pathlib.Path(output_data_location) / f"b1_{b1}-b2_{b2}-bnq_{b_nq}"
        if output_dir.exists():
            if not force:
                logger.error(f"Output directory already exists")
                logger.info(f"Use --force to overwrite")
                sys.exit(-1)
            
            if not output_dir.is_dir():
                logger.error(f"{output_dir} already exists and is not a directory")
            logger.warning(f"Writing to existing directory {output_dir.absolute()}")

        output_dir.mkdir(exist_ok=True, parents=True)

        input_file = pathlib.Path(input_data_location)
        if not input_file.exists():
            logger.error(f"Input file {input_file} not found")
            sys.exit(-2)
        try:
            checkpoint = torch.load(input_file, map_location=map_location)
        except ValueError:
            logger.error(f"Checkpoint is not a file in torch pickle format")
            sys.exit(-3)

        embedding_keys = [k for k in checkpoint.keys() if "embedding" in k]
        if len(embedding_keys) != 1:
            logger.error(
                f"{len(embedding_keys)} keys found with the word 'embedding'"
                ", use only 1 key with the name embedding"
            )
            logger.error(f"keys: {embedding_keys}")
            sys.exit(-4)

        label_keys = [k for k in checkpoint.keys() if "label" in k]
        if len(label_keys) != 1:
            logger.error(
                f"{len(label_keys)} keys found with the word 'label'"
                ", use only 1 key with the name label"
            )
            logger.error(f"keys: {label_keys}")
            sys.exit(-4)

        logger.info(sys.argv)

        data = checkpoint[embedding_keys[0]]
        labels = checkpoint[label_keys[0]]

        unique_labels = torch.unique(labels)

        for label in tqdm(unique_labels):
            idxs = labels == label
            X = data[idxs]
            logger.info(f"{X.shape}")
            assert X.shape[0] > X.shape[1]
            col_output_rank = maximum_output_rank(cr, b1, b2, b_nq, X.shape)

            X_lplr = lplr(X, col_output_rank, b1, b2, sketch=sketch, **kwargs)
            col_sketch_err = relative_tensor_error(X, X_lplr)

            X_nq = quantize(X, b_nq, preserve_original_dtype=True)
            naive_quant_err = relative_tensor_error(X, X_nq)

            logger.info(
                f"Label: {label} "
                f"Col Sketch Error: {col_sketch_err:.3f} "
                f"Dtype: {X.dtype} "
                f"Naive Quant Err: {naive_quant_err:.3f} "
            )

            torch.save(
                dict(nq=X_nq, lplr=X_lplr, label=label),
                output_dir / f"cifar10q-class={label}.pt"
            )

        if logfile is None:
            logfile = output_dir / f"quantization-{datetime.now().strftime('%m%d-%H%M%S')}.log"
        
        temp_log_file.flush()
        temp_log_file.seek(0)
        with open(logfile, "w+") as f:
            shutil.copyfileobj(temp_log_file, f)
    finally:
        temp_log_file.close()
if __name__ == "__main__":
    fire.Fire(main)
