import torch
import torch.nn as nn


MODEL_NAME = "MLP"


def create_model():
    return nn.Sequential(
        nn.Linear(768, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, 10),
    )


def create_input():
    return torch.randn(1, 768)
