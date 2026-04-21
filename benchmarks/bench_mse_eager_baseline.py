"""
Benchmark: eager MSE grid search baseline.

Works on both main and torch-compile-observers branches.
Forces exactly N grid steps (no early stopping) on a (4096, 4096) tensor
for N in 8..15.  No compile paths — pure eager timing.

Usage:
    # on main:
    git stash && git checkout main && python benchmarks/bench_mse_eager_baseline.py
    # on feature branch:
    git checkout torch-compile-observers && python benchmarks/bench_mse_eager_baseline.py
"""

import time

import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.base import Observer

# Disable compile if available (feature branch), no-op if not (main)
try:
    from llmcompressor.observers.compile_config import set_torch_compile
    set_torch_compile(False)
except ImportError:
    pass

WEIGHT_SHAPE = (4096, 4096)
QUANT_ARGS = QuantizationArgs(
    num_bits=8,
    type="int",
    symmetric=True,
    strategy="channel",
    observer="memoryless_mse",
)


def make_observer(n_steps: int) -> Observer:
    return Observer.load_from_registry(
        "memoryless_mse",
        base_name="weight",
        args=QUANT_ARGS.model_copy(),
        maxshrink=n_steps,
        grid=1.0,
        patience=n_steps + 1,
    )


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - t0) * 1000


def main():
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    weight = torch.randn(WEIGHT_SHAPE, device=device, dtype=torch.float16)

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()

    print(f"Branch:  {branch}")
    print(f"Device:  {device}  Shape: {WEIGHT_SHAPE}")
    print(f"Warmup:  {args.warmup}  Repeat: {args.repeat}")
    print()

    hdr = f"{'n':>3} {'eager (ms)':>12}"
    print(hdr)
    print("-" * len(hdr))

    for n in range(8, 16):
        obs = make_observer(n)
        for _ in range(args.warmup):
            obs(weight)
        eager_ms = sum(timed(lambda: obs(weight), device) for _ in range(args.repeat)) / args.repeat
        print(f"{n:3d} {eager_ms:12.2f}")


if __name__ == "__main__":
    main()
