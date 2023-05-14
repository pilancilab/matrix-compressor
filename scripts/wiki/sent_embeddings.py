import argparse
import pathlib
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing input lines",
    )
    args = parser.parse_args()
    return args

def main():
    
    args = parse()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not input_path.is_dir() or not input_path.exists():
        logger.error(f"--input must be the directory containing files wiki.*.stripped")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:2")
    batch_size = args.batch_size
    # Open the input file
    for filename in input_path.glob("*.stripped"):
        with open(filename, "r") as f:
            embeddings = []

            # Initialize variables for batch processing
            data = f.read().splitlines()

            for idx in tqdm(range(0, len(data), args.batch_size)):
                batch = data[idx : idx + batch_size]
                batch_embeddings: np.ndarray = model.encode(batch)
                embeddings.append(batch_embeddings)

            # Concatenate the list of embeddings into a single numpy array
            embeddings = np.concatenate(embeddings)

            if output_path.suffix == "npz":
                # Save the embeddings to a compressed numpy array file
                np.savez_compressed(output_path, embeddings=embeddings)
            else:
                np.savez_compressed(output_path / f"{filename.name}.embedding.npz", embeddings=embeddings)

if __name__ == "__main__":
    main()