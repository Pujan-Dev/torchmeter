import torch
from torch import nn

from torchmeter import benchmark


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


def test_benchmark_returns_result():
    model = TinyModel()
    inputs = torch.randn(4, 10)

    result = benchmark(
        model,
        inputs,
        device="cpu",
        warmup=2,
        iterations=10,
    )

    assert result.model_name == "PyTorch Model"
    assert result.device == "cpu"
    assert result.iterations == 10
    assert result.mean_latency_ms > 0
    assert result.median_latency_ms > 0
    assert result.p95_latency_ms > 0
    assert result.p99_latency_ms > 0
    assert result.throughput_samples_per_second > 0


def test_result_json():
    model = TinyModel()
    inputs = torch.randn(4, 10)

    result = benchmark(
        model,
        inputs,
        device="cpu",
        warmup=1,
        iterations=5,
    )

    data = result.to_dict()

    assert "mean_latency_ms" in data
    assert "throughput_samples_per_second" in data
