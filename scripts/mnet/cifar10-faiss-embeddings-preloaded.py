#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from math import ceil, floor

import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from knn.faiss_knn import FaissKNeighborsCPU, FaissKNeighborsGPU
from lplr.compressors import direct_svd_quant, lplr
from lplr.quantizers import quantize
from mnet.utils import extract_embeddings
from shared import repo_basepath
from shared.error import relative_tensor_error
import faiss

from argparse import ArgumentParser

# In[]:

parser = ArgumentParser()
parser.add_argument("--map-location", default="cpu")
args = parser.parse_known_args()

map_location = args[0].map_location
# In[]:


# Define the transform to preprocess the images
# transform = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


# In[]:


# Load the CIFAR-10 dataset
# trainset = torchvision.datasets.CIFAR10(root='/home/rsaha/varun/matrix-compressor/datasets', train=True, download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(
#     root="/home/rsaha/varun/matrix-compressor/datasets",
#     train=False,
#     download=True,
#     transform=transform,
# )

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


# In[]:


# Load the pretrained MobileNet model and remove its classification head
# model = mobilenet_v2(pretrained=True)
# model = torch.nn.Sequential(*list(model.children())[:-1])
# model.eval()


# In[]:


# Extract embeddings of the training set
# train_embeddings, train_labels = extract_embeddings(trainloader, model)

# train_embedding_checkpoint = torch.load(
#     repo_basepath
#     / "artifacts/custom_data/knn/mnet_v2/cifar10/cifar10-embeddings-mobilenet_v2.pt",
#     map_location=map_location,
# )
# train_embeddings, train_labels = (
#     train_embedding_checkpoint["train_embeddings"],
#     train_embedding_checkpoint["train_labels"],
# )


# In[]:


# Evaluate the classifier on the test set / Load test data
# test_embeddings, test_labels = extract_embeddings(testloader, model)
test_embedding_checkpoint = torch.load(
    repo_basepath
    / "artifacts/custom_data/knn/mnet_v2/cifar10/test-embeddings-labels.pt",
    map_location=map_location,
)
test_embeddings, test_labels = (
    test_embedding_checkpoint["test_embeddings"],
    test_embedding_checkpoint["test_labels"],
)

# In[]:


# knn_train = FaissKNeighborsCPU(k=5)
# logger.trace(f"Initiatinng KNN on training set")
# knn_train.fit(train_embeddings, train_labels)
# logger.trace(f"Finished KNN on training set")
# logger.trace(f"Innitiating test set prediction")
# test_predictions = knn_train.predict(test_embeddings)
# logger.trace(f"Finished test set prediction")
# print(classification_report(test_labels, test_predictions))
b_nq = 2
# In[]:
quantized_training_data = [
    torch.load(fn, map_location=map_location)
    for fn in map(
        lambda x: repo_basepath
        / f"artifacts/custom_data/knn/mnet_v2/cifar10-quantized/train/b1_8-b2_8-bnq_{b_nq}/cifar10q-class={x}.pt",
        range(10),
    )
]
train_embeddings_nq = torch.cat([qtd["nq"] for qtd in quantized_training_data])

# In[]
train_labels = torch.cat(
    [qtd["label"].tile((len(qtd["nq"]),)) for qtd in quantized_training_data]
)
train_embeddings_lplr = torch.cat([qtd["lplr"] for qtd in quantized_training_data])
# quantized_test_data = [
#     torch.load(f)
#     for f in map(
#         lambda x: f"artifacts/custom_data/knn/mnet_v2/cifar10-quantized/train/b1_8-b2_8_bnq_8/cifar10q-class={f}.pt",
#         range(10),
#     )
# ]

# In[]:

# Naive quantization and KNN accuracy


# logger.trace(f"Initiating train embedding quantization")
# train_embeddings_nq = quantize(train_embeddings, b_nq)
# logger.trace(f"Finished quantizing training embeddings")

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


print(classification_report(test_labels, test_nq_predictions))


# In[ ]:


from math import ceil, floor

from loguru import logger

from lplr.compressors import direct_svd_quant, lplr
from lplr.quantizers import quantize
from lplr.utils import maximum_output_rank
from shared.error import relative_tensor_error

# Convert to PyTorch tensor
# X_train_tensor = train_embeddings.T

# LPLR parameters
b1 = 8
b2 = 8
c = 1  # Desired compression ratio in number of parameters

# Naive quantizer parameters
nb_seed = 42

# max_dim = X_train_tensor.shape[0]
# min_dim = X_train_tensor.shape[1]
# assert max_dim >= min_dim, "Number of datapoints should be larger than dimension!"
# m = floor((c * max_dim * min_dim) / (max_dim + min_dim))
# m = maximum_output_rank(c, b1, b2, b_nq, X_train_tensor.shape)

# logger.trace(f"Starting LPLR Quant on Train Set")
# X_train_lplr = lplr(
#     X_train_tensor, m, b1, b2, seed=nb_seed, q=0, normalize_and_shift=True
# )
# logger.trace(f"Finished LPLR on train set")

# logger.trace(f"Starting Direct SVD Quant on Train set")
# X_train_dir_svd = direct_svd_quant(X_train_tensor, r=m, B1=b1, B2=b2)
# logger.trace(f"Finished Direct SVD Quant")

# # X_train_nq = quantize(X_train_tensor, b_nq)
# X_train_nq = train_embeddings_nq

# logger.info(f"X_train.shape: {X_train_tensor.shape}")
# logger.info(
#     f"LPLR accuracy b1 = {b1}, b2 = {b2}, compression-ratio: {c}, sketch size = {m}, fro-norm error: {relative_tensor_error(X_train_tensor, X_train_lplr):.3f}"
# )
# logger.info(
#     f"Direct-SVD quant. accuracy b1 = {b1}, b2 = {b2}, sketch size = {m}, fro-norm error: {relative_tensor_error(X_train_tensor, X_train_dir_svd):.3f}"
# )
# logger.info(
#     f"Naive quant. accuracy b = {b_nq}, fro-norm error: {relative_tensor_error(X_train_tensor, X_train_nq):.3f}"
# )


# In[ ]:

X_train_lplr = train_embeddings_lplr
# Train a nearest neighbors classifier using the naively quantized embeddings
knn_lplr = FaissKNeighborsCPU(3)
logger.trace(f"Initiating fit on LPLR array")
knn_lplr.fit(X_train_lplr, train_labels)
logger.trace(f"finished fit on LPLR array")

# Evaluate the classifier on the test set

test_lplr_predictions = knn_lplr.predict(test_embeddings)

print(classification_report(test_labels, test_lplr_predictions))

# %%
