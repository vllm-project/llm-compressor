"""
Benchmark: eager vs torch.compiled MSE observer grid search.

Measures wall-clock time for a fixed number of grid search steps (N) on a
(4096, 4096) weight tensor (Llama-3-8B qkvo_proj, channel W8).  Early stopping
is disabled by setting patience > N so the loop always runs exactly N steps in
eager mode and ceil(N/chunk)*chunk steps in compiled mode.

Usage:
    python benchmarks/bench_mse_compile.py [--device cuda] [--warmup 3] [--repeat 10]
"""

import argparse
import math
import time
from typing import List

import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.base import Observer
from llmcompressor.observers.compile_config import set_torch_compile
from llmcompressor.observers.helpers import flatten_for_calibration


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


def make_weight(device: torch.device) -> torch.Tensor:
    w = torch.randn(WEIGHT_SHAPE, device=device, dtype=torch.float16)
    rows = WEIGHT_SHAPE[0]
    ch_scale = torch.empty(rows, 1, device=device, dtype=torch.float16).log_normal_(
        mean=0.0, std=0.5
    )
    w = w * ch_scale
    n_outlier = max(1, rows // 100)
    idx = torch.randperm(rows, device=device)[:n_outlier]
    w[idx] *= torch.empty(n_outlier, 1, device=device, dtype=torch.float16).uniform_(
        5.0, 10.0
    )
    return w


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(
    n_steps: int,
    chunk_sizes: List[int],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict:
    weight = make_weight(device)
    results = {"n_steps": n_steps}

    # --- Eager ---
    set_torch_compile(False)
    obs = make_observer(n_steps)
    for _ in range(warmup):
        obs(weight)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        obs(weight)
    sync(device)
    eager_ms = (time.perf_counter() - t0) / repeat * 1000
    results["eager_ms"] = eager_ms

    # --- Compiled for each chunk_size ---
    for cs in chunk_sizes:
        compiled_steps = math.ceil(n_steps / cs) * cs
        set_torch_compile(True, chunk_size=cs)

        # Cold compile
        torch._dynamo.reset()
        obs_cold = make_observer(n_steps)
        sync(device)
        t0 = time.perf_counter()
        obs_cold(weight)
        sync(device)
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm compile (cache hit, new observer + weight)
        weight2 = make_weight(device)
        obs_warm = make_observer(n_steps)
        sync(device)
        t0 = time.perf_counter()
        obs_warm(weight2)
        sync(device)
        warm_ms = (time.perf_counter() - t0) * 1000

        # Steady-state
        for _ in range(warmup):
            obs_warm(weight2)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(repeat):
            obs_warm(weight2)
        sync(device)
        run_ms = (time.perf_counter() - t0) / repeat * 1000

        saving = eager_ms - run_ms
        if saving > 0:
            be_cold = math.ceil((cold_ms - run_ms) / saving)
            be_warm = math.ceil((warm_ms - run_ms) / saving)
        else:
            be_cold = float("inf")
            be_warm = float("inf")

        results[f"cs{cs}"] = {
            "compiled_steps": compiled_steps,
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "run_ms": run_ms,
            "speedup": eager_ms / run_ms if run_ms > 0 else 0,
            "be_cold": be_cold,
            "be_warm": be_warm,
        }

    set_torch_compile(False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MSE observer compile path")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--n-steps", type=int, nargs="+", default=[5, 10, 20],
    )
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=[1, 3, 5, 7],
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Device:      {device}")
    print(f"Shape:       {WEIGHT_SHAPE}")
    print(f"Warmup:      {args.warmup}   Repeat: {args.repeat}")
    print(f"N-steps:     {args.n_steps}")
    print(f"Chunk sizes: {args.chunk_sizes}")
    print()

    all_results = []
    for n in args.n_steps:
        print(f"=== n_steps={n} (eager runs {n}, compiled runs ceil(n/cs)*cs) ===")
        r = bench(n, args.chunk_sizes, device, args.warmup, args.repeat)
        all_results.append(r)

        print(f"  eager: {r['eager_ms']:8.2f} ms  ({n} steps)")
        for cs in args.chunk_sizes:
            d = r[f"cs{cs}"]
            be_c = f"{d['be_cold']}" if d["be_cold"] != float("inf") else "never"
            be_w = f"{d['be_warm']}" if d["be_warm"] != float("inf") else "never"
            print(
                f"  cs={cs}:  {d['run_ms']:8.2f} ms  ({d['compiled_steps']:>3d} steps)  "
                f"cold={d['cold_ms']:.0f}ms  warm={d['warm_ms']:.0f}ms  "
                f"speedup={d['speedup']:.2f}x  "
                f"breakeven: {be_c} cold / {be_w} warm"
            )
        print()

    # Summary table: rows = (n_steps, chunk_size), one table per n_steps
    for r in all_results:
        n = r["n_steps"]
        print(f"--- n_steps={n}  eager={r['eager_ms']:.2f}ms ---")
        print(
            f"  {'cs':>4} {'steps':>5} {'run ms':>8} {'speedup':>8} "
            f"{'cold ms':>8} {'warm ms':>8} {'BE cold':>8} {'BE warm':>8}"
        )
        for cs in args.chunk_sizes:
            d = r[f"cs{cs}"]
            be_c = f"{d['be_cold']}" if d["be_cold"] != float("inf") else "never"
            be_w = f"{d['be_warm']}" if d["be_warm"] != float("inf") else "never"
            print(
                f"  {cs:4d} {d['compiled_steps']:5d} {d['run_ms']:8.2f} "
                f"{d['speedup']:8.2f}x {d['cold_ms']:8.0f} {d['warm_ms']:8.0f} "
                f"{be_c:>8} {be_w:>8}"
            )
        print()


if __name__ == "__main__":
    main()
