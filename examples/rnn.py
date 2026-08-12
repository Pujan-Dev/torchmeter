import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.classifier(output[:, -1, :])


MODEL_NAME = "Simple RNN"


def create_model():
    return SimpleRNN()


def create_input():
    return torch.randn(8, 100, 128)
