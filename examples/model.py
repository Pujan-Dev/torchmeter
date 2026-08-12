import torch
import torchvision.models as models


MODEL_NAME = "ResNet50"

def create_model():
    return models.resnet50(weights=None)


def create_input():
    return torch.randn(1, 3, 224, 224)
