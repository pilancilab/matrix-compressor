import pathlib
import torch
from shared import repo_basepath


def lname_to_input_path(lname: str, index_file: str):
    import json

    with open(index_file) as f:
        mapping = json.load(f)
    return mapping["weight_map"][lname]


def load_layer(
    lname,
    base_path=repo_basepath / "artifacts/llama-7b-hf",
    index_file="pytorch_model.bin.index.json",
    map_location="cuda:0",
):
    base_path = pathlib.Path(base_path)
    in_path = lname_to_input_path(lname, base_path / index_file)
    return torch.load(base_path / in_path, map_location=map_location)[lname]
