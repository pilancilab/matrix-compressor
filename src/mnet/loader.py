import torch
import timm
from typing import Literal

def load_mobilenet_without_head(version: Literal[2, 3] = 3):
    assert version in (2,3)
    
    if version == 2:
        model_name = 'mobilenetv2_100'
    elif version == 3:
        model_name = "mobilenetv3_large_100"
    else:
        raise ValueError("version must be 2 or 3")
    model = timm.create_model(model_name, pretrained=True)
    return model