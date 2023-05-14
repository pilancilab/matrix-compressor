#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from math import ceil, floor
import pathlib
import sys
from typing import Final

import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger

from torchvision.models import vit_b_16, ViT_B_16_Weights

from mnet.utils import extract_embeddings
from shared import repo_basepath

from argparse import ArgumentParser

# In[]:

parser = ArgumentParser()
parser.add_argument("--map-location", default="cpu")
parser.add_argument(
    "-o",
    "--out-path",
    type=str,
    required=True,
    help="Directory to store embeddings in",
)
args = parser.parse_known_args()

map_location = args[0].map_location
out_path = repo_basepath / args[0].out_path
model_name: Final = "vit_b_16"

if out_path.exists():
    confirm = input("Output path already exists, continue? (y/n)")
    while confirm not in ("y","n"):
        logger.error(f"Please input (y/n)")
        confirm = input("Output path already exists, continue? (y/n)")
    if confirm == "n":
        sys.exit(-1)
        

# In[]:


# Define the transform to preprocess the images
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# In[]
# Function to extract embeddings from the dataset using MobileNet
def extract_embeddings(dataloader, model):
    from tqdm import tqdm
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, targets = data
            features = model(images)
            features = features.view(features.size(0), -1)
            embeddings.append(features)
            labels.append(targets)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels

# In[]:


# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root="/home/rsaha/varun/matrix-compressor/artifacts/datasets",
    train=True,
    download=True,
    transform=transform,
)
testset = torchvision.datasets.CIFAR10(
    root="/home/rsaha/varun/matrix-compressor/artifacts/datasets",
    train=False,
    download=True,
    transform=transform,
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


# In[]:


# Load the pretrained ViT model and remove its classification head
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
# model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


# In[]:


# Extract embeddings of the training set
train_embeddings, train_labels = extract_embeddings(trainloader, model)

logger.info(f"train_embeddings.shape: {train_embeddings.shape}")
logger.info(f"type(train_embeddings): {type(train_embeddings)}")
logger.info(f"train_labels.shape: {train_labels.shape}")

torch.save(
    {"train_embeddings": train_embeddings, "train_labels": train_labels},
    out_path / "cifar10-train-embeddings.pt",
)
