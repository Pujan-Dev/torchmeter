import torch
import torch.nn as nn


MODEL_NAME = "Simple CNN"


def create_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def create_input():
    return torch.randn(1, 3, 224, 224)
