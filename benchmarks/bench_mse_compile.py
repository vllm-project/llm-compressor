"""
Benchmark: eager vs compiled MSE grid search on a (4096, 4096) tensor.

Forces exactly N grid steps (no early stopping) for N in 8..15.
Compiled path runs ceil(N/chunk_size)*chunk_size steps.
Reports eager time, cold compile, warm compile, steady-state, and breakeven.

Usage:
    python benchmarks/bench_mse_compile.py [--device cuda] [--warmup 3] [--repeat 10]
"""

import math
import time

import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.base import Observer
from llmcompressor.observers.compile_config import set_torch_compile

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    cs = args.chunk_size
    weight = torch.randn(WEIGHT_SHAPE, device=device, dtype=torch.float16)

    print(f"Device: {device}  Shape: {WEIGHT_SHAPE}  Chunk size: {cs}")
    print(f"Warmup: {args.warmup}  Repeat: {args.repeat}")
    print()

    results = []

    for n in range(8, 16):
        compiled_steps = math.ceil(n / cs) * cs

        # --- Eager ---
        set_torch_compile(False)
        obs_e = make_observer(n)
        for _ in range(args.warmup):
            obs_e(weight)
        eager_ms = sum(timed(lambda: obs_e(weight), device) for _ in range(args.repeat)) / args.repeat

        # --- Compiled: cold (fresh dynamo cache) ---
        set_torch_compile(True, chunk_size=cs)
        torch._dynamo.reset()
        obs_cold = make_observer(n)
        cold_ms = timed(lambda: obs_cold(weight), device)

        # --- Compiled: warm (cache populated, new observer + weight) ---
        weight2 = torch.randn(WEIGHT_SHAPE, device=device, dtype=torch.float16)
        obs_warm = make_observer(n)
        warm_ms = timed(lambda: obs_warm(weight2), device)

        # --- Compiled: steady-state ---
        for _ in range(args.warmup):
            obs_warm(weight2)
        run_ms = sum(timed(lambda: obs_warm(weight2), device) for _ in range(args.repeat)) / args.repeat

        # --- Breakeven ---
        # N_cold: cold_ms + (N-1)*run_ms = N*eager_ms  =>  N = (cold - run) / (eager - run)
        # N_warm: warm_ms + (N-1)*run_ms = N*eager_ms  =>  N = (warm - run) / (eager - run)
        saving = eager_ms - run_ms
        if saving > 0:
            be_cold = math.ceil((cold_ms - run_ms) / saving)
            be_warm = math.ceil((warm_ms - run_ms) / saving)
        else:
            be_cold = float("inf")
            be_warm = float("inf")

        speedup = eager_ms / run_ms if run_ms > 0 else 0
        results.append((n, compiled_steps, eager_ms, cold_ms, warm_ms, run_ms, speedup, be_cold, be_warm))

    set_torch_compile(False)

    # --- Print ---
    hdr = f"{'n':>3} {'comp':>4} {'eager':>9} {'cold':>9} {'warm':>9} {'run':>9} {'speedup':>8} {'BE cold':>8} {'BE warm':>8}"
    units = f"{'':>3} {'steps':>4} {'(ms)':>9} {'(ms)':>9} {'(ms)':>9} {'(ms)':>9} {'(x)':>8} {'(reps)':>8} {'(reps)':>8}"
    print(hdr)
    print(units)
    print("-" * len(hdr))
    for n, cs_steps, eager, cold, warm, run, spd, be_c, be_w in results:
        be_c_s = f"{be_c}" if be_c != float("inf") else "never"
        be_w_s = f"{be_w}" if be_w != float("inf") else "never"
        print(
            f"{n:3d} {cs_steps:4d} {eager:9.2f} {cold:9.1f} {warm:9.1f} "
            f"{run:9.2f} {spd:8.2f}x {be_c_s:>8} {be_w_s:>8}"
        )


if __name__ == "__main__":
    main()
