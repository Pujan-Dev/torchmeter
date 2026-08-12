import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


MODEL_NAME = "LSTM"


def create_model():
    return LSTMModel()


def create_input():
    return torch.randn(8, 100, 128)
