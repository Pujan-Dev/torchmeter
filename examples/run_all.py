import subprocess
import sys


MODELS = [
    "examples/mlp.py",
    "examples/cnn.py",
    "examples/rnn.py",
    "examples/lstm.py",
    "examples/transformer.py",
    "examples/resnet.py",
]


for model in MODELS:
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchmeter",
            "run",
            model,
            "--device",
            "cuda",
        ],
        check=False,
    )

    if result.returncode != 0:
        print(f"FAILED: {model}")
