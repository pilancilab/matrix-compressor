import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")


# Function to generate sentence embeddings
def generate_sentence_embedding(text):
    # Tokenize the input text
    input_ids = tokenizer.encode(
        text, add_special_tokens=True, max_length=128, truncation=True
    )

    # Convert the tokenized input into a PyTorch tensor
    input_ids = torch.tensor([input_ids])

    # Get the BERT output
    with torch.no_grad():
        output = model(input_ids)[0][:, 0, :]

    # Convert the output to a numpy array
    output_array = output.numpy()

    # Return the numpy array
    return output_array


def main():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    args = parser.parse_args()

    # Open the input file
    with open(args.input, "r") as f:
        embeddings = []

        # Loop over each line in the file
        for line in f:
            # Remove any leading or trailing whitespace from the line
            line = line.strip()

            # Generate the sentence embedding for the line
            embedding = generate_sentence_embedding(line)

            # Append the embedding to the list of embeddings
            embeddings.append(embedding)

        # Convert the list of embeddings to a numpy array
        embeddings = np.array(embeddings)

        # Save the embeddings to a compressed numpy array file
        np.savez_compressed(args.output, embeddings=embeddings)


if __name__ == "__main__":
    main()
