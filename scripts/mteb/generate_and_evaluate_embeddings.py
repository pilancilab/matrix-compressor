#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
from multiprocessing import freeze_support
import os
import pathlib
import sys
from argparse import ArgumentParser
from math import ceil, floor
from typing import Final

import numpy as np
from sklearn.metrics import classification_report
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from loguru import logger
from sentence_transformers import SentenceTransformer
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm

from knn.faiss_knn import FaissKNeighborsCPU
from lplr.compressors import Sketch, lplr
from lplr.quantizers import quantize
from lplr.utils import maximum_output_rank
from mnet.utils import extract_embeddings
from shared import repo_basepath
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

# %%

def embed(model, dataset, split="train", batch_size=1024, return_label=True,device=torch.device("cuda"), pool=None):
    data = dataset[split]
    
    sentences = data["text"]
    labels = torch.tensor(data["label"])
    
    logger.trace(f"{len(sentences)} sentences")
    logger.trace(f"{len(labels)} labels")
    embeddings = []
    for idx in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[idx : idx + batch_size]
        if pool is not None:
            batch_embeddings: np.ndarray = model.encode_multi_process(batch, pool=pool, batch_size=batch_size)
        else:
            batch_embeddings: np.ndarray = model.encode(batch, batch_size=batch_size, device=device)
        # logger.trace(f"{len(batch_embeddings)} sentences encode to size {batch_embeddings.shape}")
        embeddings.append(batch_embeddings)
    
    embeddings = np.concatenate(embeddings)
    logger.trace(f"Embeddings {embeddings.shape}")
    if return_label:
         return embeddings, labels
    return embeddings
# In[]


def parse():
    parser = ArgumentParser()
    # Generic
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
    
    # Embedding args
    parser.add_argument("--model-device", default="cuda:2")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing input lines",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", choices=["all-MiniLM-L6-v2","all-mpnet-base-v2","all-MiniLM-L12-v2","multi-qa-mpnet-base-dot-v1","all-distilroberta-v1","multi-qa-MiniLM-L6-cos-v1"], help="Sentence Transformer model to use for embedding")
    parser.add_argument("--dataset", default="mteb/imdb", help="Dataset to use for classifications")
    parser.add_argument("--use-multiprocessing",action="store_true")
    return parser.parse_known_args()[0]


# In[]
# def main():
freeze_support()
args = parse()

log_file = f"misc/{args.dataset}/{args.model}/eval-{datetime.now().strftime('%m%d-%H%M%S')}.log"
pathlib.Path(log_file).parent.mkdir(exist_ok=True, parents=True)
logger.add(sink=open(log_file, "w+"), level="DEBUG")
logger.debug(f"Parsed input arguments {args}")
device = torch.device(args.map_location)
multiprocessing = args.use_multiprocessing

# Embedding args
model = args.model
dataset = args.dataset
model_device = args.model_device
batch_size = args.batch_size

ds = load_dataset(dataset)
model = SentenceTransformer(args.model, device=model_device)

pool=None
if multiprocessing:
    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()


logger.trace(f"Embedding train dataset")
train_data, train_labels = embed(model, ds, "train", batch_size=batch_size, device=model_device, pool=pool)
logger.info(f"Train Data Text: {train_data.shape} Label: {train_labels.shape}")
logger.trace(f"Finished train embeddings")

logger.trace(f"Embedding test dataset")
test_data, test_labels = embed(model, ds, "test", batch_size=batch_size, device=model_device, pool=pool)
logger.info(f"Test Data Text: {test_data.shape} Label: {test_labels.shape}")
logger.trace(f"Finished test embeddding")

if multiprocessing:
    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

assert test_data.shape[0] == len(test_labels)
assert train_data.shape[0] == len(train_labels)

# Sketching Args
input_dir = pathlib.Path(args.input)
cr = args.cr
b1 = args.b1
b2 = args.b2
b_nq = args.b_nq
sketch = args.sketch
k = args.num_neighbors
sparse_jl_s = args.sparse_jl_s


# %%
train_tensor = torch.from_numpy(train_data).to(device)
train_labels = torch.tensor(train_labels)
X = train_tensor


logger.debug(
    f"Train tensor loaded with shape = {train_tensor.shape} on device {device}"
)

test_tensor = torch.from_numpy(test_data).to(device)
test_labels = torch.tensor(test_labels)
logger.debug(f"Test tensor loaded with shape = {test_tensor.shape} on device {device}")

assert train_tensor.shape[0] > train_tensor.shape[1]


# %%

# KNN Train

knn_tr = FaissKNeighborsCPU(k)
logger.trace(f"Initiating train KNN fit")
knn_tr.fit(X, train_labels)
logger.trace(f"Finished KNN fit on train")

# tr_dists, tr_ind = knn_tr.predict_neighbors(test_tensor)
test_tr_predictions = knn_tr.predict(test_tensor)
logger.info(classification_report(test_labels, test_tr_predictions))

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
knn_nq.fit(X_nq, train_labels)
logger.trace(f"Finished KNN fit on naive quant")

# nq_dists, nq_ind = knn_nq.predict_neighbors(test_tensor)
# logger.info(f"NQ IOU: {evaluate(tr_ind, nq_ind)}")
# logger.info(f"NQ Dists {evaluate_distances(nq_ind, X, test_tensor)}")
test_nq_predictions = knn_nq.predict(test_tensor)
logger.info(classification_report(test_labels, test_nq_predictions))


# %%
# KNN LPLR

# LPLR
col_output_rank = maximum_output_rank(cr, b1, b2, b_nq, X.shape)
logger.debug(f"Initiating LPLR with params = {b1} {b2} {b_nq} {cr}")

X_lplr = lplr(X, col_output_rank, b1, b2, sketch=sketch, sparse_jl_s=sparse_jl_s)
col_sketch_err = relative_tensor_error(X, X_lplr)
logger.debug(f"Finished LPLR with col sketch error = {col_sketch_err:.3f}")

# Train a nearest neighbors classifier using the LPLR quantized embeddings
knn_lplr = FaissKNeighborsCPU(k)
logger.trace(f"Initiating fit on LPLR array")
knn_lplr.fit(X_lplr, train_labels)
logger.trace(f"finished fit on LPLR array")

# lplr_dists, lplr_ind = knn_lplr.predict_neighbors(test_tensor)
# logger.info(f"LPLR IOU: {evaluate(tr_ind, lplr_ind)}")
# logger.info(f"LPLR Dists {evaluate_distances(lplr_ind, X, test_tensor)}")

test_lplr_predictions = knn_lplr.predict(test_tensor)
logger.info(classification_report(test_labels, test_lplr_predictions))
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
# %%

logger.info(
    f"Col Sketch Error: {col_sketch_err:.3f} "
    # f"Row Sketch Error: {row_sketch_err:.3f} "
    f"Dtype: {X.dtype} "
    f"Naive Quant Err: {naive_quant_err:.3f} "
)
