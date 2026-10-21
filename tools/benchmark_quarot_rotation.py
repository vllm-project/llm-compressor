# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only dense/structured QuaRot benchmark on synthetic GLM-sized weights.

Each variant runs in a fresh process. Timings include rotate_axis allocations and
storage conversion; construction is timed separately. Sampled RSS includes the
runtime/allocator and may miss short peaks. No checkpoint or NPU is used.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch

from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)


@torch.no_grad()
def measure(args):
    torch.set_num_threads(args.threads)
    storage = getattr(torch, args.storage)
    value = (
        torch.randn(args.rows, args.size, generator=torch.Generator().manual_seed(71))
        * 0.02
    ).to(storage)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        started = time.perf_counter()
        structured = make_hadamard_rotation(
            args.size,
            block_size=args.block_size,
            shifted=args.shifted,
            dtype=torch.float32,
        )
        rotation = structured.to_dense() if args.worker == "dense" else structured
        construction = time.perf_counter() - started
        options = dict(axis=1, precision=torch.float32)
        del_output = rotate_axis(value, rotation, **options)  # Warmup.
        del del_output
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = rotate_axis(value, rotation, **options)
            samples.append(time.perf_counter() - started)
            del result
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    report = {
        "variant": args.worker,
        "construction_seconds": construction,
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "rotation_storage_bytes": (
            rotation.numel() * rotation.element_size()
            if args.worker == "dense"
            else structured.block.numel() * structured.block.element_size()
        ),
        "arithmetic_segment_bytes": value.numel() * 4,
        "baseline_rss_bytes": baseline,
        "sampled_peak_rss_bytes": peak[0],
        "sampled_peak_rss_increase_bytes": peak[0] - baseline,
    }
    if args.worker == "structured":
        # Accuracy outside both timing and memory measurements. to_dense reproduces
        # the original B or B @ P @ B construction in FP32.
        dense = structured.to_dense()
        expected = rotate_axis(value, dense, **options)
        actual = rotate_axis(value, structured, **options)
        delta = actual.double() - expected.double()
        report["max_abs_diff"] = delta.abs().max().item()
        report["relative_l2"] = (delta.norm() / expected.double().norm()).item()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--size", type=int, default=6144)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--storage", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--worker", choices=("dense", "structured"), help=argparse.SUPPRESS
    )
    parser.add_argument("--shifted", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rows, args.size, args.block_size, args.threads, args.repeats) <= 0:
        parser.error("Dimensions, threads and repeats must be positive")
    if args.worker:
        report = measure(args)
    else:
        report = {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "shape": [args.rows, args.size],
            "axis": 1,
            "block_size": args.block_size,
            "storage": args.storage,
            "precision": "float32",
            "threads": args.threads,
            "warmup": 1,
            "repeats": args.repeats,
            "memory_scope": (
                "2ms sampled RSS in fresh processes, includes construction; "
                "accuracy excluded"
            ),
            "cases": [],
        }
        with tempfile.TemporaryDirectory(prefix="quarot-benchmark-") as temporary:
            for shifted in (False, True):
                case = {"shifted": shifted}
                for variant in ("dense", "structured"):
                    output = Path(temporary) / f"{variant}.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        variant,
                        "--output",
                        str(output),
                    ]
                    for name in (
                        "rows",
                        "size",
                        "block_size",
                        "threads",
                        "repeats",
                        "storage",
                    ):
                        command.extend(
                            ["--" + name.replace("_", "-"), str(getattr(args, name))]
                        )
                    if shifted:
                        command.append("--shifted")
                    subprocess.run(command, check=True)
                    case[variant] = json.loads(output.read_text(encoding="utf-8"))
                case["speedup"] = (
                    case["dense"]["median_seconds"]
                    / case["structured"]["median_seconds"]
                )
                report["cases"].append(case)
        print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
