#!/usr/bin/env python
# coding: utf-8

"""Usage

❱ B1=8 B2=8 BNQ=8 python scripts/mnetv3/cifar10_faiss_embeddings_preloaded.py --quant-directory artifacts/custom_data/knn/mnetv3/cifar10-
quantized/train/b1_$B1-b2_$B2-bnq_$BNQ --b1 $B1 --b2 $B2 --b_nq $BNQ --train-dir artifacts/custom_data/knn/mnetv3/cifar10 --eval-train
"""
# In[1]:


from argparse import ArgumentParser
from typing import Final

import torch
import torchvision.transforms as transforms
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report

from knn.faiss_knn import FaissKNeighborsCPU
from shared import repo_basepath
from shared.error import relative_tensor_error

# In[]:

parser = ArgumentParser()
parser.add_argument("--map-location", default="cpu")
parser.add_argument("--quant-directory")
parser.add_argument("--b1", type=int, default=8)
parser.add_argument("--b2", type=int, default=8)
parser.add_argument("--b_nq", type=int, default=1)
parser.add_argument("--eval-train", action="store_true")
parser.add_argument("--train-dir")
args = parser.parse_known_args()

known_args = args[0]
map_location = known_args.map_location
b1: Final = known_args.b1
b2: Final = known_args.b2
b_nq: Final = known_args.b_nq
quant_dir: Final = known_args.quant_directory
train_dir: Final = known_args.train_dir
# In[]:

test_embedding_checkpoint = torch.load(
    repo_basepath / f"{train_dir}/cifar10-test-embeddings.pt",
    map_location=map_location,
)
test_embeddings, test_labels = (
    test_embedding_checkpoint["test_embeddings"],
    test_embedding_checkpoint["test_labels"],
)

# In[]:
quantized_training_data: Final = [
    torch.load(fn, map_location=map_location)
    for fn in map(
        lambda x: repo_basepath / f"{quant_dir}/cifar10q-class={x}.pt",
        range(10),
    )
]
train_embeddings_nq: Final = torch.cat([qtd["nq"] for qtd in quantized_training_data])

# In[]
train_labels: Final = torch.cat(
    [qtd["label"].tile((len(qtd["nq"]),)) for qtd in quantized_training_data]
)
train_embeddings_lplr: Final = torch.cat(
    [qtd["lplr"] for qtd in quantized_training_data]
)


# In[]

if known_args.eval_train:
    train_embedding_checkpoint = torch.load(
        repo_basepath / f"{known_args.train_dir}/cifar10-train-embeddings.pt",
        map_location=map_location,
    )
    train_embeddings, train_labels_orig = (
        train_embedding_checkpoint["train_embeddings"],
        train_embedding_checkpoint["train_labels"],
    )

    # InSanity checks
    # print(f"Fro Norm Error: {relative_tensor_error(train_embeddings, train_embeddings_lplr)}")
    # print(f"Fro Norm Error: {relative_tensor_error(train_embeddings, train_embeddings_nq)}")

    knn_tr = FaissKNeighborsCPU(3)
    logger.trace(f"Initiating train KNN fit")
    knn_tr.fit(train_embeddings, train_labels_orig)
    logger.trace(f"Finished KNN fit on train")

    # Evaluate the classifier on the test set
    test_predictions = knn_tr.predict(test_embeddings)
    logger.trace(f"Finished predictions on test embeddings")

    print(f"Baseline Results")
    print(classification_report(test_labels, test_predictions))

# In[]:
# Train a nearest neighbors classifier using the naively quantized embeddings
knn_nq = FaissKNeighborsCPU(3)
logger.trace(f"Initiating naive quant KNN fit")
knn_nq.fit(train_embeddings_nq, train_labels)
logger.trace(f"Finished KNN fit on naive quant")

# In[ ]:


# Evaluate the classifier on the test set
test_nq_predictions = knn_nq.predict(test_embeddings)
logger.trace(f"Finished predictions on test embeddings")

# In[ ]:

print(f"Naive Quantization Results with b_nq = {b_nq}")
print(classification_report(test_labels, test_nq_predictions))


# In[ ]:
# LPLR parameters
c = 1  # Desired compression ratio in number of parameters
# Naive quantizer parameters
nb_seed = 42

# In[ ]:

X_train_lplr = train_embeddings_lplr
# Train a nearest neighbors classifier using the naively quantized embeddings
knn_lplr = FaissKNeighborsCPU(3)
logger.trace(f"Initiating fit on LPLR array")
knn_lplr.fit(X_train_lplr, train_labels)
logger.trace(f"finished fit on LPLR array")

# Evaluate the classifier on the test set

test_lplr_predictions = knn_lplr.predict(test_embeddings)

print(f"LPLR Results with b_1 = {b1} b_2 = {b2} b_nq = {b_nq} CR = {c}")
print(classification_report(test_labels, test_lplr_predictions))
