# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Pytest imports this conftest before test modules under ``tests/lmeval/``.
``transformers`` caches torchvision availability at first import, so we ensure
``torchvision`` is importable before ``test_lmeval`` pulls in Hugging Face.
"""

import os
import subprocess
import sys


def _bootstrap_torchvision_before_transformers() -> None:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        pass
    else:
        return

    if os.environ.get("LLMCOMPRESSOR_DISABLE_TORCHVISION_PIP"):
        print(
            "WARNING: torchvision is not installed; hf-multimodal LM Eval tests "
            "will fail until it is available. Install torchvision in the image or "
            "unset LLMCOMPRESSOR_DISABLE_TORCHVISION_PIP to allow a pip install at "
            "startup.",
            file=sys.stderr,
        )
        return

    print(
        "torchvision not found; running `pip install torchvision` before importing "
        "lm_eval/transformers",
        file=sys.stderr,
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "torchvision"],
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "torchvision is required for hf-multimodal LM Eval tests but "
            f"`pip install torchvision` failed ({exc}). Install torchvision in the "
            "test image using the same PyTorch/CUDA channel as torch."
        ) from exc
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torchvision still not importable after pip install. "
            "Install a build that matches your PyTorch/CUDA stack."
        ) from exc


_bootstrap_torchvision_before_transformers()
