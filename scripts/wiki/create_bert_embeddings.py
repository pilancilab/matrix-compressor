import argparse
import pathlib
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Function to generate sentence embeddings
def generate_sentence_embeddings(texts, model, tokenizer, device=torch.device("cuda"), max_length = 512):
    # Tokenize the input texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)

    # Get the BERT output
    with torch.no_grad():
        output = model(**encoded_input)[0][:, 0, :]

    # Convert the output to a numpy array
    output_array = output.detach().to("cpu").numpy()

    # Return the numpy array
    return output_array


def parse():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input directory"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing input lines",
    )
    parser.add_argument("--model", default="google/bert_uncased_L-4_H-256_A-4", help="Sentence Transformer model to use for embedding")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of token in a sentence to truncate to")
    args = parser.parse_args()
    return args


def main():
    
    args = parse()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    model = args.model
    
    if not input_path.is_dir() or not input_path.exists():
        logger.error(f"--input must be the directory containing files wiki.*.stripped")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    batch_size = args.batch_size
    max_length = args.max_tokens
    # Load the BERT Google tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    # Set the maximum length limit to 128
    
    tokenizer.padding_side = "right" # Padding will be applied on the right side of the sequence

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and tokenizer to the GPU
    model.to(device)
    
    
    # Open the input file
    for filename in input_path.glob("*.stripped"):
        with open(filename, "r") as f:
            embeddings = []

            # Initialize variables for batch processing
            data = f.read().splitlines()

            for idx in tqdm(range(0, len(data), args.batch_size)):
                batch = data[idx : idx + batch_size]
                batch_embeddings: np.ndarray = generate_sentence_embeddings(batch, model, tokenizer, device=device, max_length=max_length)
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
