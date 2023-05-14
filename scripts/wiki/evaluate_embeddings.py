#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from math import ceil, floor
import pathlib
import sys
from typing import Final
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from knn.faiss_knn import FaissKNeighborsCPU
from lplr.compressors import Sketch, lplr
from lplr.quantizers import quantize
from lplr.utils import maximum_output_rank

from mnet.utils import extract_embeddings
from shared import repo_basepath

from argparse import ArgumentParser

from shared.error import relative_tensor_error

# %%


def evaluate_indices_numpy(x1: np.ndarray, x2: np.ndarray):
    intersection = np.intersect1d(x1, x2)
    union = np.union1d(x1, x2)
    return intersection.size / union.size


def evaluate_indices_list(x1, x2):
    x1 = set(x1)
    x2 = set(x2)

    return len(x1.intersection(x2)) / len(x1.union(x2))


def evaluate_distances(indices, target, test):
    dists = []
    for ind_row, trow in zip(indices, test):
        tvecs = target[ind_row]
        dists.append(
            torch.mean(torch.tensor([torch.linalg.norm(xx - trow) for xx in tvecs]))
        )
    return np.mean(dists)


def evaluate(X1, X2):
    assert X1.dtype == np.dtype("int64")
    assert X2.dtype == np.dtype("int64")

    assert X1.shape == X2.shape

    ious = []
    for x1, x2 in zip(X1.tolist(), X2.tolist()):
        ious.append(evaluate_indices_list(x1, x2))

    return np.mean(ious)


# In[]


def parse():
    parser = ArgumentParser()
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="artifacts/wikipedia/wikitext-2-raw",
        help="Path to input directory",
    )
    parser.add_argument(
        "--b1",
        type=int,
        default=8,
        help="bit budget for Q1/right factor quantizer in LPLR",
    )
    parser.add_argument("--b2", default=8, type=int, help="bit budget for Q2 in LPLR")
    parser.add_argument(
        "--b_nq", default=1, type=int, help="bit budget for naive quantizer"
    )
    parser.add_argument(
        "--cr", default=1, type=float, help="Compression ratio for lplr"
    )
    parser.add_argument(
        "--sketch",
        choices=list(map(lambda x: x.value, Sketch)),
        default=Sketch.GAUSSIAN,
    )
    parser.add_argument(
        "-k",
        "--num-neighbors",
        type=int,
        default=3,
        help="Number of nearest neighbors in KNN",
    )
    parser.add_argument(
        "--sparse-jl-s",
        type=int,
        default=1,
        help="sparsity factory for Sparse JL Transform",
    )
    return parser.parse_known_args()[0]


# In[]
# def main():
args = parse()
logger.trace(f"Parsed input arguments {args}")
input_dir = pathlib.Path(args.input)
device = torch.device(args.map_location)
cr = args.cr
b1 = args.b1
b2 = args.b2
b_nq = args.b_nq
sketch = args.sketch
k = args.num_neighbors
sparse_jl_s = args.sparse_jl_s

# %%
train_tensor = torch.from_numpy(
    np.load(input_dir / "wiki.train.raw.stripped.embedding.npz")["embeddings"],
).to(device)
X = train_tensor


logger.trace(
    f"Train tensor loaded with shape = {train_tensor.shape} on device {device}"
)

test_tensor = torch.from_numpy(
    np.load(input_dir / "wiki.test.raw.stripped.embedding.npz")["embeddings"],
).to(device)
logger.trace(f"Test tensor loaded with shape = {test_tensor.shape} on device {device}")

assert train_tensor.shape[0] > train_tensor.shape[1]


# %%

# KNN Train

knn_tr = FaissKNeighborsCPU(k)
logger.trace(f"Initiating train KNN fit")
knn_tr.fit(X, torch.empty(X.shape[0]))
logger.trace(f"Finished KNN fit on train")

tr_dists, tr_ind = knn_tr.predict_neighbors(test_tensor)

# %%
# KNN NQ

# NQ
logger.trace(f"Initiating naive quantization with b_nq = {b_nq}")
X_nq = quantize(X, b_nq, preserve_original_dtype=True)
naive_quant_err = relative_tensor_error(X, X_nq)
logger.trace(f"Finished NQ with error = {naive_quant_err:.3f}")


# Train a nearest neighbors classifier using the naively quantized embeddings
knn_nq = FaissKNeighborsCPU(k)
logger.trace(f"Initiating naive quant KNN fit")
knn_nq.fit(X_nq, torch.empty(X_nq.shape[0]))
logger.trace(f"Finished KNN fit on naive quant")

nq_dists, nq_ind = knn_nq.predict_neighbors(test_tensor)
logger.info(f"NQ IOU: {evaluate(tr_ind, nq_ind)}")
logger.info(f"NQ Dists {evaluate_distances(nq_ind, X, test_tensor)}")

# %%
# KNN LPLR

# LPLR
col_output_rank = maximum_output_rank(cr, b1, b2, b_nq, X.shape)
logger.trace(f"Initiating LPLR with params = {b1} {b2} {b_nq} {cr}")

X_lplr = lplr(X, col_output_rank, b1, b2, sketch=sketch, sparse_jl_s=sparse_jl_s)
col_sketch_err = relative_tensor_error(X, X_lplr)
logger.trace(f"Finished LPLR with col sketch error = {col_sketch_err:.3f}")

# Train a nearest neighbors classifier using the LPLR quantized embeddings
knn_lplr = FaissKNeighborsCPU(k)
logger.trace(f"Initiating fit on LPLR array")
knn_lplr.fit(X_lplr, torch.empty(X_lplr.shape[0]))
logger.trace(f"finished fit on LPLR array")

lplr_dists, lplr_ind = knn_lplr.predict_neighbors(test_tensor)

logger.info(f"LPLR IOU: {evaluate(tr_ind, lplr_ind)}")
logger.info(f"LPLR Dists {evaluate_distances(lplr_ind, X, test_tensor)}")


# %%
# # KNN LPLR ROW

# row_output_rank = maximum_output_rank(cr, b1, b2, b_nq, X.T.shape)
# logger.trace(f"Initiating LPLR with params = {b1} {b2} {b_nq} {cr}")

# X_lplr_row = lplr(X.T, row_output_rank, b1, b2, sketch=sketch, sparse_jl_s=sparse_jl_s)
# row_sketch_err = relative_tensor_error(X.T, X_lplr_row)
# logger.trace(f"Finished LPLR with row sketch error = {row_sketch_err:.3f}")

# # Train a nearest neighbors classifier using the LPLR quantized embeddings
# knn_lplr_row = FaissKNeighborsCPU(k)
# logger.trace(f"Initiating fit on LPLR array")
# knn_lplr_row.fit(X_lplr_row.T, torch.empty(X_lplr_row.T.shape[0]))
# logger.trace(f"finished fit on LPLR array")

# lplr_dists_row, lplr_ind_row = knn_lplr.predict_neighbors(test_tensor)

# logger.info(f"LPLR IOU: {evaluate(tr_ind, lplr_ind_row)}")
# logger.info(f"LPLR Dists {evaluate_distances(lplr_ind_row, X, test_tensor)}")


logger.info(
    f"Col Sketch Error: {col_sketch_err:.3f} "
    # f"Row Sketch Error: {row_sketch_err:.3f} "
    f"Dtype: {X.dtype} "
    f"Naive Quant Err: {naive_quant_err:.3f} "
)
