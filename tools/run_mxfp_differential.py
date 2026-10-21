# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict synthetic MXFP parity probe. Exit 1 means numerical differences exist."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.mxfp.differential import run_matrix  # noqa: E402
from tests.mxfp.oracle import MxOracle  # noqa: E402
from tools.run_quarot_checks import git_info  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelslim-source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    oracle = MxOracle(args.modelslim_source.resolve())
    report = run_matrix(oracle)
    report.update(
        real_weights=False,
        elapsed_seconds=time.perf_counter() - started,
        llmc=git_info(REPO),
        modelslim=git_info(args.modelslim_source),
        scope=(
            "Finite synthetic group-32 last-axis minmax weights/dynamic activations; "
            "original ModelSlim observer, weight driver, Q/DQ and linear forward "
            "vs LLMC/CT"
        ),
        excludes=[
            "NPU kernels",
            "checkpoint byte layouts",
            "real GLM accuracy",
            "nonfinite input policy",
        ],
        reference_source_sha256=oracle.hashes,
        compressed_tensors_source_sha256={
            name: hashlib.sha256(
                Path(
                    importlib.metadata.distribution("compressed-tensors").locate_file(
                        name
                    )
                ).read_bytes()
            ).hexdigest()
            for name in (
                "compressed_tensors/quantization/quant_args.py",
                "compressed_tensors/quantization/quant_scheme.py",
                "compressed_tensors/quantization/utils/mxfp_utils.py",
                "compressed_tensors/quantization/utils/helpers.py",
                "compressed_tensors/quantization/lifecycle/forward.py",
                "compressed_tensors/quantization/lifecycle/forward_helpers.py",
            )
        },
        llmc_source_sha256={
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "src/llmcompressor/observers/base.py",
                "src/llmcompressor/observers/min_max.py",
                "src/llmcompressor/observers/helpers.py",
                "src/llmcompressor/modifiers/quantization/quantization/base.py",
                "src/llmcompressor/modifiers/quantization/quantization/mixin.py",
                "tests/quarot/source_loader.py",
            )
        },
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "compressed-tensors", "transformers")
        },
        harness_source_sha256={
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *REPO.joinpath("tests/mxfp").glob("*.py")]
        },
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / "mxfp-parity.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "parity_passed",
                    "tensor_cases",
                    "tensor_cases_with_differences",
                    "linear_cases",
                    "linear_cases_with_differences",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    print(f"Report: {output}")
    return 0 if report["parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
