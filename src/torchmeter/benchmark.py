from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from .result import BenchmarkResult


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA so GPU operations are fully completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        raise ValueError("Cannot calculate percentile from empty data.")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return values[lower] + (values[upper] - values[lower]) * weight

def _set_batch_size(inputs: object, batch_size: int) -> object:
    """Create inputs with the requested batch size."""
    if torch.is_tensor(inputs):
        if inputs.ndim == 0:
            return inputs

        shape = list(inputs.shape)
        shape[0] = batch_size

        return torch.randn(
            shape,
            dtype=inputs.dtype,
            device=inputs.device,
        )

    if isinstance(inputs, dict):
        return {
            key: _set_batch_size(value, batch_size)
            for key, value in inputs.items()
        }

    if isinstance(inputs, tuple):
        return tuple(
            _set_batch_size(value, batch_size)
            for value in inputs
        )

    if isinstance(inputs, list):
        return [
            _set_batch_size(value, batch_size)
            for value in inputs
        ]

    return inputs
def _run_model(
    model: torch.nn.Module,
    inputs: object,
) -> object:
    """Run the model with the provided inputs."""
    if isinstance(inputs, dict):
        return model(**inputs)

    if isinstance(inputs, (tuple, list)):
        return model(*inputs)

    return model(inputs)


def _move_to_device(
    value: object,
    device: torch.device,
) -> object:
    """Move tensors or nested structures to the target device."""
    if torch.is_tensor(value):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]

    return value


def benchmark(
    model: torch.nn.Module,
    inputs: object,
    *,
    model_name: str = "PyTorch Model",
    device: str = "auto",
    warmup: int = 20,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark a PyTorch model.

    Args:
        model: PyTorch model to benchmark.
        inputs: Example model input.
        model_name: Human-readable model name.
        device: "auto", "cpu", or "cuda".
        warmup: Number of warmup iterations.
        iterations: Number of measured iterations.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_device = torch.device(device)

    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = model.to(target_device)
    model.eval()

    inputs = _move_to_device(inputs, target_device)

    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    with torch.inference_mode():
        for _ in range(warmup):
            _run_model(model, inputs)

        _synchronize(target_device)

        timings: list[float] = []

        for _ in range(iterations):
            _synchronize(target_device)

            start = time.perf_counter()

            _run_model(model, inputs)

            _synchronize(target_device)

            end = time.perf_counter()

            timings.append((end - start) * 1000)

    mean_latency = statistics.mean(timings)
    median_latency = statistics.median(timings)

    p95 = _percentile(timings, 0.95)
    p99 = _percentile(timings, 0.99)

    throughput = 1000.0 / mean_latency

    peak_memory = None

    if target_device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated(target_device)
            / (1024**2)
        )

    return BenchmarkResult(
        model_name=model_name,
        device=str(target_device),
        iterations=iterations,
        warmup_iterations=warmup,
        mean_latency_ms=mean_latency,
        median_latency_ms=median_latency,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        throughput_samples_per_second=throughput,
        peak_memory_mb=peak_memory,
    )
