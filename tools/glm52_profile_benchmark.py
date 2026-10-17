# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only profiler overhead measurement, not a model or accelerator benchmark."""

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from examples.glm52_precision.oneshot_profile import Profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expected = value @ value.T
    rng = torch.get_rng_state().clone()
    report = {
        "scope": "CPU host overhead only; no NPU/8-rank speedup claim",
        "iterations": args.iterations,
        "repeats": args.repeats,
        "modes": {},
    }
    for mode in ("off", "summary", "sampled", "full"):
        durations = []
        for _ in range(args.repeats):
            profile = Profile("cpu", mode if mode != "off" else "summary")
            started = time.perf_counter()
            for _ in range(args.iterations):
                with (
                    nullcontext()
                    if mode == "off"
                    else profile.stage("forward", hot=True)
                ):
                    actual = value @ value.T
            durations.append(time.perf_counter() - started)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), rng)
        report["modes"][mode] = {
            "seconds": durations,
            "median_seconds": statistics.median(durations),
            "profile": profile.report,
            "result_exact": True,
            "rng_unchanged": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {mode: value["median_seconds"] for mode, value in report["modes"].items()}
        )
    )


if __name__ == "__main__":
    main()
