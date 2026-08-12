# TorchMeter

Simple, reproducible performance benchmarking for PyTorch models.

TorchMeter makes it easy to measure the performance of PyTorch models and generate useful benchmark results from the command line.

## Status

🚧 **Early Development — v0.1.0**

TorchMeter is currently an experimental open-source project.

The API and CLI may change as the project develops.

## Features

- PyTorch model benchmarking
- CPU benchmarking
- CUDA benchmarking
- Warmup iterations
- Configurable benchmark iterations
- Mean latency
- Median latency
- P95 latency
- P99 latency
- Throughput measurement
- Peak CUDA memory measurement
- JSON output
- Markdown reports
- Simple Python API
- Reproducible environment information

## Installation

### Using uv

```bash
uv add torchmeter
```

### From source

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/torchmeter.git
cd torchmeter
```

Install the development environment:

```bash
uv sync
```

## Quick Start

Create a Python file containing a model and example input.

For example:

```python
import torch
import torchvision.models as models


MODEL_NAME = "ResNet50"


def create_model():
    return models.resnet50(weights=None)


def create_input():
    return torch.randn(1, 3, 224, 224)
```

Save it as:

```text
examples/model.py
```

Run the benchmark:

```bash
uv run torchmeter run examples/model.py
```

For CUDA:

```bash
uv run torchmeter run examples/model.py --device cuda
```

For CPU:

```bash
uv run torchmeter run examples/model.py --device cpu
```

## Model Interface

TorchMeter expects the model file to provide two functions:

```python
def create_model():
    ...


def create_input():
    ...
```

You can optionally provide a model name:

```python
MODEL_NAME = "My Model"
```

If `MODEL_NAME` is not provided, TorchMeter uses the model class name.

## Example Models

The repository includes several example models:

```text
examples/
├── cnn.py
├── lstm.py
├── mlp.py
├── model.py
├── rnn.py
├── resnet.py
└── transformer.py
```

Run an example:

```bash
uv run torchmeter run examples/cnn.py --device cuda
```

```bash
uv run torchmeter run examples/rnn.py --device cuda
```

```bash
uv run torchmeter run examples/lstm.py --device cuda
```

```bash
uv run torchmeter run examples/transformer.py --device cuda
```

```bash
uv run torchmeter run examples/resnet.py --device cuda
```

## Benchmark Configuration

Control the number of warmup iterations:

```bash
uv run torchmeter run examples/model.py --warmup 20
```

Control the number of benchmark iterations:

```bash
uv run torchmeter run examples/model.py --iterations 100
```

You can combine options:

```bash
uv run torchmeter run examples/model.py \
    --device cuda \
    --warmup 20 \
    --iterations 100
```

## Output Formats

### Terminal

The default output is a human-readable benchmark:

```text
TorchMeter Benchmark
────────────────────────────────
Model:       ResNet50
Device:      cuda
Iterations:  100

Latency
  Mean:       4.823 ms
  Median:     4.761 ms
  P95:        5.132 ms
  P99:        5.421 ms

Throughput:  207.34 samples/sec
Peak memory: 1420.31 MB
```

### JSON

Generate machine-readable output:

```bash
uv run torchmeter run examples/model.py \
    --format json
```

Write the JSON result to a file:

```bash
uv run torchmeter run examples/model.py \
    --format json \
    --output results.json
```

### Markdown

Generate a Markdown report:

```bash
uv run torchmeter run examples/model.py \
    --format markdown
```

Write it to a file:

```bash
uv run torchmeter run examples/model.py \
    --format markdown \
    --output benchmark.md
```

## Python API

TorchMeter can also be used directly from Python.

```python
import torch

from torchmeter import benchmark


model = torch.nn.Sequential(
    torch.nn.Linear(768, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10),
)

inputs = torch.randn(1, 768)

result = benchmark(
    model,
    inputs,
    device="cuda",
    warmup=20,
    iterations=100,
)

print(result.mean_latency_ms)
print(result.throughput_samples_per_second)
```

## Benchmark Results

A benchmark contains:

- Model name
- Device
- Number of warmup iterations
- Number of benchmark iterations
- Mean latency
- Median latency
- P95 latency
- P99 latency
- Throughput
- Peak CUDA memory when available

## Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run the test suite with verbose output:

```bash
uv run pytest -v
```

Run Ruff:

```bash
uv run ruff check .
```

Format the project:

```bash
uv run ruff format .
```

## Project Structure

```text
torchmeter/
├── examples/
│   ├── cnn.py
│   ├── lstm.py
│   ├── mlp.py
│   ├── model.py
│   ├── rnn.py
│   ├── resnet.py
│   └── transformer.py
│
├── src/
│   └── torchmeter/
│       ├── __init__.py
│       ├── __main__.py
│       ├── benchmark.py
│       ├── cli.py
│       └── result.py
│
├── tests/
│   └── test_benchmark.py
│
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

## Design Goals

TorchMeter aims to be:

1. **Simple** — benchmark a model with a single command.
2. **Reproducible** — capture the environment used for benchmarking.
3. **Accurate** — correctly measure CPU and CUDA workloads.
4. **Scriptable** — provide JSON output for automation.
5. **Extensible** — provide both a CLI and Python API.
6. **Open source** — developed transparently with community contributions.

## Roadmap

### v0.1.x

- [x] Basic PyTorch benchmarking
- [x] CPU support
- [x] CUDA support
- [x] Latency statistics
- [x] Throughput measurement
- [x] CUDA memory measurement
- [x] JSON output
- [x] Markdown output
- [x] Python API
- [x] Example models
- [x] Basic tests

### Future

Potential future features include:

- `torch.compile` comparisons
- Performance regression detection
- Benchmark baselines
- CI integration
- GitHub Actions
- Additional hardware metrics
- Distributed benchmarking
- Benchmark history
- Performance dashboards

Future features are subject to change based on project feedback and development priorities.

## Contributing

Contributions are welcome.

Before opening a pull request, please run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

For larger changes, please open an issue first to discuss the proposed approach.

## License

TorchMeter is licensed under the Apache License 2.0.

See [LICENSE](LICENSE) for the full license text.

## Disclaimer

TorchMeter is an early-stage project. Benchmark results can vary depending on hardware, software versions, system load, and benchmark configuration.
Always compare results under controlled conditions when making performance decisions.
