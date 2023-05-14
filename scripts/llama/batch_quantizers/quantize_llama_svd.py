#!/usr/bin/env python
# coding: utf-8

from typing import Literal
from loguru import logger

import torch
import torch.nn as nn
from llama import hf_load
from lplr.compressors import direct_svd_quant
import fire
import pathlib
from tqdm import tqdm
import re

    

def main(
    ckpt_dir: pathlib.Path = "./artifacts/LLAMA/llama-7b-hf",
    out_dir: pathlib.Path = "./artifacts/LLAMA/llama-quantized",
    b1: int = 16,
    b2: int = 16,
    f: float = 0.7,
):
    generator = hf_load(ckpt_dir, skip_tokenizer=True)
    model = generator.model
    logger.debug(f"Quantizing model using B1 = {b1} and B2 = {b2}")
    
    with torch.no_grad():
        from math import ceil, floor

        for name, layer in tqdm(model.named_parameters()):
            param = layer.data.detach()
            if param.ndim == 2:
                layer.data = direct_svd_quant(param, f=f, B1=b1, B2=b2)
            logger.debug(
                f"Quantizing layer: {name} with shape {layer.shape} on device {layer.device} and original dtype: {layer.dtype}"
            )
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving quantized model to {out_dir}")
    model.save_pretrained(out_dir)


if __name__ == "__main__":
    fire.Fire(main)
