import argparse
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Load the BERT Tiny tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
# Set the maximum length limit to 128
max_length = 512
tokenizer.padding_side = "right" # Padding will be applied on the right side of the sequence

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model and tokenizer to the GPU
model.to(device)
# tokenizer.to(device)

# Function to generate sentence embeddings
def generate_sentence_embeddings(texts):
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
    batch_size = args.batch_size
    # Open the input file
    with open(args.input, "r") as f:
        embeddings = []

        # Initialize variables for batch processing
        data = f.read().splitlines()

        for idx in tqdm(range(0, len(data), args.batch_size)):
            batch = data[idx : idx + batch_size]
            batch_embeddings: np.ndarray = generate_sentence_embeddings(batch)
            embeddings.append(batch_embeddings)

        # Concatenate the list of embeddings into a single numpy array
        embeddings = np.concatenate(embeddings)

        # Save the embeddings to a compressed numpy array file
        np.savez_compressed(args.output, embeddings=embeddings)


if __name__ == "__main__":
    main()
