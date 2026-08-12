"""TorchMeter: PyTorch model performance benchmarking."""

from .benchmark import benchmark
from .result import BenchmarkResult

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "benchmark",
]