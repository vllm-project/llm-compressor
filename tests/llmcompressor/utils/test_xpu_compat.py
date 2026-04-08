# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fake-XPU tests for the torch.accelerator migration.

Two layers of testing:

1. Unit tests (no GPU): mock torch.accelerator to report "xpu" and verify
   that device-selection helpers produce the right torch.device objects.

2. E2E smoke test (requires GPU): combine DeviceRemapMode (xpu->cuda at the
   C++ boundary) with a torch.accelerator mock so that the full oneshot()
   data-free FP8 pipeline runs thinking it is on XPU while real CUDA hardware
   executes the kernels.

   The e2e test exercises:
   - get_main_device() returning xpu:0
   - gpu_if_available() returning xpu:0
   - datafree CalibrationPipeline (no calibration data)
   - QuantizationModifier FP8 weight quantization
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.entrypoints.model_free.helpers import gpu_if_available
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils.dev import get_main_device
from tests.emulate_device import DeviceRemapMode
from tests.testing_utils import requires_gpu

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

FAKE_TYPE = "xpu"


@pytest.fixture
def mock_xpu_accelerator(monkeypatch):
    """Mock torch.accelerator to report XPU as the current device (no GPU needed).

    current_accelerator() returns a torch.device in the real API, so we mock
    it with torch.device("xpu") rather than SimpleNamespace to match the
    contract that callers rely on (e.g. torch.device(current_accelerator(), idx)).
    """
    fake = torch.device(FAKE_TYPE)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 1)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)


# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_main_device_fake_xpu(mock_xpu_accelerator):
    """get_main_device() should return xpu:0 when the accelerator reports XPU."""
    device = get_main_device()
    assert device == torch.device("xpu", 0), f"Expected xpu:0, got {device}"


@pytest.mark.unit
def test_get_main_device_cpu_fallback(monkeypatch):
    """get_main_device() should fall back to CPU when no accelerator is available."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    device = get_main_device()
    assert device.type == "cpu", f"Expected cpu, got {device}"


@pytest.mark.unit
def test_gpu_if_available_fake_xpu(mock_xpu_accelerator):
    """gpu_if_available(None) should return xpu:0 when accelerator reports XPU."""
    device = gpu_if_available(None)
    assert device == torch.device("xpu", 0), f"Expected xpu:0, got {device}"


@pytest.mark.unit
def test_gpu_if_available_explicit_device(mock_xpu_accelerator):
    """gpu_if_available with an explicit device should ignore the accelerator state."""
    device = gpu_if_available("cpu")
    assert device == torch.device("cpu"), f"Expected cpu, got {device}"


@pytest.mark.unit
def test_gpu_if_available_cpu_fallback(monkeypatch):
    """gpu_if_available(None) should return cpu when no accelerator is available."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    device = gpu_if_available(None)
    assert device.type == "cpu", f"Expected cpu, got {device}"


# ---------------------------------------------------------------------------
# E2E smoke test (requires a real CUDA GPU)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.smoke
def test_oneshot_datafree_fp8_fake_xpu(tmp_path):
    """
    Full oneshot() data-free FP8 quantization under emulated XPU.

    Two-layer emulation:
    - DeviceRemapMode: intercepts all torch.* calls, silently rewrites
      xpu->cuda so real CUDA hardware executes the kernels.
    - torch.accelerator mock: Python-level code (get_main_device,
      gpu_if_available, metric_logging, etc.) sees "xpu" as the
      current accelerator type.

    Verifies that the migrated torch.accelerator call sites produce a
    valid quantized model on non-CUDA device types.
    """
    real_type = torch.accelerator.current_accelerator().type  # "cuda"
    real_device_count = torch.accelerator.device_count()
    real_is_available = torch.accelerator.is_available()

    orig_current_accelerator = torch.accelerator.current_accelerator
    orig_device_count = torch.accelerator.device_count
    orig_is_available = torch.accelerator.is_available
    orig_current_device_index = torch.accelerator.current_device_index

    fake = torch.device(FAKE_TYPE)
    torch.accelerator.current_accelerator = lambda: fake
    torch.accelerator.device_count = lambda: real_device_count
    torch.accelerator.is_available = lambda: real_is_available
    torch.accelerator.current_device_index = lambda: 0

    try:
        with DeviceRemapMode(fake_type=FAKE_TYPE, real_type=real_type):
            model = AutoModelForCausalLM.from_pretrained(
                "nm-testing/tinysmokellama-3.2",
                dtype=torch.float16,
                device_map=f"{FAKE_TYPE}:0",
            )

            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8",
                ignore=["lm_head"],
            )

            model = oneshot(
                model=model,
                recipe=recipe,
                output_dir=str(tmp_path / "xpu_fp8_out"),
            )

        # Verify at least one linear layer was quantized
        quantized = [
            name
            for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear)
            and name != "lm_head"
            and hasattr(mod, "weight_scale")
        ]
        assert len(quantized) > 0, "Expected at least one quantized Linear layer"

    finally:
        torch.accelerator.current_accelerator = orig_current_accelerator
        torch.accelerator.device_count = orig_device_count
        torch.accelerator.is_available = orig_is_available
        torch.accelerator.current_device_index = orig_current_device_index
