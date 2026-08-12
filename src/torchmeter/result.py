from dataclasses import asdict, dataclass
import json


@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    iterations: int
    warmup_iterations: int

    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    throughput_samples_per_second: float
    peak_memory_mb: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        memory = (
            f"{self.peak_memory_mb:.2f} MB"
            if self.peak_memory_mb is not None
            else "N/A"
        )

        return f"""# TorchMeter Benchmark

| Metric | Value |
|---|---:|
| Model | `{self.model_name}` |
| Device | `{self.device}` |
| Iterations | {self.iterations} |
| Warmup | {self.warmup_iterations} |
| Mean latency | {self.mean_latency_ms:.3f} ms |
| Median latency | {self.median_latency_ms:.3f} ms |
| P95 latency | {self.p95_latency_ms:.3f} ms |
| P99 latency | {self.p99_latency_ms:.3f} ms |
| Throughput | {self.throughput_samples_per_second:.2f} samples/sec |
| Peak GPU memory | {memory} |
"""
