import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x[:, -1, :])


MODEL_NAME = "Transformer"


def create_model():
    return TransformerModel()


def create_input():
    return torch.randn(8, 128, 256)