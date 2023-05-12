# Load pre-trained Llama LLM

# MASTER_PORT=1234 MASTER_ADDR=127.0.0.1 RANK=0 CUDA_VISIBLE_DEVICES=2 LOCAL_RANK=0 WORLD_SIZE=1

from typing import Tuple
import os
import torch
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from .fb_llama import ModelArgs, Transformer, Tokenizer, LLaMA

from loguru import logger


def setup_model_parallel(
    local_rank=None,
    world_size=None,
    master_addr="127.0.0.1",
    master_port="1234",
    seed=42,
) -> Tuple[int, int]:
    if not local_rank:
        logger.trace(f"Setting local rank internally to 0")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    if not world_size:
        logger.trace(f"Setting world size internally to 1")
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["WORLD_SIZE"] = str(world_size)
    if not os.environ.get("MASTER_ADDR"):
        logger.trace(f"Setting master address to {master_addr}")
        os.environ["MASTER_ADDR"] = master_addr
    if not os.environ.get("MASTER_PORT"):
        logger.trace(f"Setting master port manually to {master_port}")
        os.environ["MASTER_PORT"] = master_port

    logger.info(f"local_rank={local_rank}, world_size={world_size}, rank={local_rank}")

    torch.distributed.init_process_group("nccl", world_size=world_size, rank=local_rank)
    initialize_model_parallel(world_size)

    # seed must be the same in all processes
    torch.manual_seed(seed)
    return local_rank, world_size


def fb_load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    logger.debug(f"ckpt_dir:{ckpt_dir}")
    logger.info(f"checkpoints:{checkpoints}")
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[0]
    logger.trace("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    logger.trace(tokenizer)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def hf_load(path: str, device="auto", skip_tokenizer=False) -> LLaMA:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.trace("Loading CausalLM Model")
    model = AutoModelForCausalLM.from_pretrained(path, device_map=device)
    logger.trace("Loaded CausalLM model")
    
    if skip_tokenizer:
        return model
    
    logger.trace("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(path, device_map=device)
    logger.trace("Loaded Tokenizer")
    
    
    generator = LLaMA(model, tokenizer)
    return generator
