from loguru import logger

from llama import fb_load, setup_model_parallel


def main():
    setup_model_parallel()
    ckpt_dir = "/home/rsaha/LLaMA-7B/"
    tokenizer_path = "/home/rsaha/LLaMA-7B/tokenizer.model"
    max_seq_len: int = 512
    max_batch_size: int = 32

    generator = fb_load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    for name, layer in generator.model.named_parameters():
        logger.debug(f"Layer: {name} has shape {layer.shape}")


if __name__ == "__main__":
    main()
