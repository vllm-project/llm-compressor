# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the checkpoint-free QuaRot L0–L3 gate and write inspectable evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def git_info(path):
    def read(*args):
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "status": read("status", "--porcelain", "--untracked-files=normal"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelslim-source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--include-flex-smooth",
        action="store_true",
        help="Also require FlexSmooth L0-L3 and QuaRot composition",
    )
    args = parser.parse_args()
    reference = args.modelslim_source.resolve()
    if not (reference / "msmodelslim/model/glm_5/quarot.py").is_file():
        parser.error("--modelslim-source must be a local ModelSlim source checkout")
    repo = Path(__file__).resolve().parents[1]
    output = args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    xml = output / "junit.xml"
    env = os.environ.copy()
    env.update(
        MODELSLIM_SOURCE=str(reference),
        QUAROT_REPORT_DIR=str(output),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/quarot",
        "-q",
        "-p",
        "no:cacheprovider",
        "--require-quarot-reference",
        f"--junitxml={xml}",
    ]
    if args.include_flex_smooth:
        command.extend(
            [
                "tests/flex_smooth",
                "tests/llmcompressor/transformers/compression/test_resave_config.py",
                "tests/llmcompressor/modeling/test_linear_experts.py",
                "tests/llmcompressor/pipelines/sequential/test_ast_helpers.py",
                "tests/llmcompressor/pipelines/sequential/ast_utils.py/test_auto_wrapper.py",
            ]
        )
        env.update(FLEXSMOOTH_REQUIRE_REFERENCE="1", FLEXSMOOTH_REPORT_DIR=str(output))
    started = time.perf_counter()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "pytest", "numpy", "compressed-tensors")
    }
    # Allocate a unique directory owned by this run; pytest clears its basetemp.
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=output) as temporary:
        command.append(f"--basetemp={temporary}")
        completed = subprocess.run(command, cwd=repo, env=env)
    counts = {}
    if xml.is_file():
        suites = ET.parse(xml).getroot().iter("testsuite")
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
    passed = (
        completed.returncode == 0
        and counts.get("tests", 0) > 0
        and not any(counts.get(key, 0) for key in ("failures", "errors", "skipped"))
    )
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
        "scope": (
            "Power-of-two non-MTP QuaRot L0-L3: source differential, public lifecycle, "
            "CPU/disk offload, official tiny GLM oneshot and local reload; "
            "NOT real-model verification"
        )
        + (
            "; also FlexSmooth norm-linear/OV L0-L3, basic/sequential linear mixed "
            "MXFP4/MXFP8 composition, FP32/BF16 compressed roundtrip with scoped GLM "
            "expert construction and explicit dtype normalization, cache-aligned "
            "CPU/disk offload, config persistence and header-only server fixtures; "
            "two-rank shared CPU/disk sequential oneshot and compressed reload; "
            "NOT generic loader compatibility, server kernels, KV-cache or real L4"
            if args.include_flex_smooth
            else ""
        ),
        "python": sys.version,
        "dependencies": versions,
        "llmc": git_info(repo),
        "modelslim": git_info(reference),
        "counts": counts,
        "command": command,
        "tested_source_sha256": {
            str(path.relative_to(repo)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(
                [
                    *repo.joinpath("src/llmcompressor/modifiers/transform/quarot").glob(
                        "*.py"
                    ),
                    *repo.joinpath("tests/quarot").glob("*.py"),
                    repo / "src/llmcompressor/modeling/fuse.py",
                    repo / "src/llmcompressor/modifiers/transform/__init__.py",
                    Path(__file__).resolve(),
                    *(
                        [
                            *repo.joinpath(
                                "src/llmcompressor/modifiers/transform/flex_smooth"
                            ).glob("*.py"),
                            *repo.joinpath("tests/flex_smooth").glob("*.py"),
                            repo / "examples/glm52_precision/mixed_mxfp.yaml",
                            repo / "examples/glm52_precision/oneshot_profile.py",
                            repo / "src/llmcompressor/modeling/moe/linear_experts.py",
                            repo
                            / "src/llmcompressor/modifiers/transform/utils"
                            / "distributed.py",
                            repo
                            / "src/llmcompressor/pipelines/sequential"
                            / "ast_utils/auto_wrapper.py",
                            repo
                            / "tests/llmcompressor/modeling/test_linear_experts.py",
                            repo
                            / "tests/llmcompressor/pipelines/sequential"
                            / "test_ast_helpers.py",
                            repo
                            / "tests/llmcompressor/pipelines/sequential"
                            / "ast_utils.py/test_auto_wrapper.py",
                            repo / "tools/glm52_precision_probe.py",
                            repo
                            / "src/llmcompressor/transformers/compression"
                            / "compressed_tensors_utils.py",
                            repo
                            / "tests/llmcompressor/transformers/compression"
                            / "test_resave_config.py",
                        ]
                        if args.include_flex_smooth
                        else []
                    ),
                ]
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"passed": passed, "counts": counts, "report_dir": str(output)}, indent=2
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
