from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .benchmark import benchmark


def _load_module(path: Path):
    """Load a Python file as a module."""
    spec = importlib.util.spec_from_file_location(
        "torchmeter_user_model",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(path.parent))

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchmeter",
        description="Performance benchmarking for PyTorch models.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Benchmark a Python model file.",
    )

    run_parser.add_argument(
        "model",
        type=Path,
        help="Path to the model.py file.",
    )

    run_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    run_parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    run_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    run_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
    )

    run_parser.add_argument(
        "--output",
        type=Path,
        help="Write the result to a file.",
    )
    run_parser.add_argument(
        "--batch-sizes",
        type=str,
        help="Comma-separated batch sizes to benchmark, e.g. 1,8,16,32.",
    )
    return parser


def _print_text(result) -> None:
    print()
    print("TorchMeter Benchmark")
    print("─" * 32)
    print(f"Model:       {result.model_name}")
    print(f"Device:      {result.device}")
    print(f"Iterations:  {result.iterations}")
    print()
    print("Latency")
    print(f"  Mean:       {result.mean_latency_ms:.3f} ms")
    print(f"  Median:     {result.median_latency_ms:.3f} ms")
    print(f"  P95:        {result.p95_latency_ms:.3f} ms")
    print(f"  P99:        {result.p99_latency_ms:.3f} ms")
    print()
    print(f"Throughput:  {result.throughput_samples_per_second:.2f} samples/sec")

    if result.peak_memory_mb is not None:
        print(f"Peak memory: {result.peak_memory_mb:.2f} MB")

    print()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    if not args.model.exists():
        parser.error(f"Model file does not exist: {args.model}")

    try:
        module = _load_module(args.model)

        if not hasattr(module, "create_model"):
            parser.error("model.py must define create_model().")

        if not hasattr(module, "create_input"):
            parser.error("model.py must define create_input().")

        model = module.create_model()
        inputs = module.create_input()

        model_name = getattr(
            module,
            "MODEL_NAME",
            model.__class__.__name__,
        )

        result = benchmark(
            model,
            inputs,
            model_name=model_name,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    except Exception as exc:
        parser.error(str(exc))

    if args.format == "json":
        output = result.to_json()
    elif args.format == "markdown":
        output = result.to_markdown()
    else:
        _print_text(result)
        output = None

    if output is not None:
        if args.output:
            args.output.write_text(output + "\n")
            print(f"Report written to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
