from loguru import logger

import torch
import torch.nn as nn
from llama import hf_load, setup_model_parallel


def main():
    torch.set_printoptions(precision=3)
    ckpt_dir = "./artifacts/LLAMA/llama-7b-hf"

    generator = hf_load(ckpt_dir)
    # param_dict = {}
    for name, layer in generator.model.named_parameters():
        param = layer.data.detach()
        rank = -1
        if param.ndim == 2:
            rank = torch.linalg.matrix_rank(param)
            # svd_vals = torch.linalg.svdvals(param)
            # param_dict[name] = svd_vals
        logger.debug(f"Layer: {name} has shape {layer.shape} with rank {rank} with dtype: {layer.dtype}")
        
        
    # torch.save(param_dict, "misc/llama/llama_singular_vals.pt")


if __name__ == "__main__":
    main()
