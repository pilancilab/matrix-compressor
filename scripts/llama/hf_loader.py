from loguru import logger

from llama_compression import hf_load, setup_model_parallel


def main():
    ckpt_dir = "./artifacts/llama-7b-hf"

    generator = hf_load(ckpt_dir)

    for name, layer in generator.model.named_parameters():
        logger.debug(f"Layer: {name} has shape {layer.shape}")


if __name__ == "__main__":
    main()
