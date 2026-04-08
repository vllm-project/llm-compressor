# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the torch.accelerator migration.

Mock torch.accelerator to report "xpu" and verify that device-selection
helpers produce the right torch.device objects. No GPU required.
"""

import pytest
import torch

from llmcompressor.entrypoints.model_free.helpers import gpu_if_available
from llmcompressor.utils.dev import get_main_device

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
