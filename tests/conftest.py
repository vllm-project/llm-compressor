# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Top-level conftest for ``--emulate-xpu`` device emulation.

When ``--emulate-xpu`` is passed, three layers of patching are activated
**before** test collection so that the entire test suite runs under
emulated XPU identity on real CUDA hardware:

  1. DeviceRemapMode — intercepts torch.* calls, remaps "xpu" -> "cuda"
  2. Accelerator mock — torch.accelerator.current_accelerator() reports "xpu"
  3. is_accelerator_type patch — accepts both "xpu" and "cuda"
"""

import torch


def pytest_addoption(parser):
    parser.addoption(
        "--emulate-xpu",
        action="store_true",
        default=False,
        help="Emulate XPU device identity on CUDA hardware via TorchFunctionMode",
    )


def pytest_configure(config):
    """Activate device emulation before test collection.

    Three layers of patching:
      1. DeviceRemapMode — intercepts torch.* functions, remaps "xpu" -> "cuda"
      2. Accelerator mock — torch.accelerator.current_accelerator() reports "xpu"
      3. is_accelerator_type patch — accepts both "xpu" and "cuda"

    Layer 3 is necessary because DeviceRemapMode converts torch.device("xpu") ->
    torch.device("cuda"), so tensor.device.type is "cuda".  But is_accelerator_type
    compares against the mocked "xpu" and would return False.
    """
    if not config.getoption("--emulate-xpu"):
        return

    from tests.emulate_device import DeviceRemapMode

    real_type = torch.accelerator.current_accelerator().type  # "cuda"
    fake_type = "xpu"

    # Save originals for cleanup
    config._emulate_orig_current_accelerator = torch.accelerator.current_accelerator
    config._emulate_orig_device_count = torch.accelerator.device_count
    config._emulate_orig_is_available = torch.accelerator.is_available
    config._emulate_orig_current_device_index = torch.accelerator.current_device_index

    # Snapshot real values before mocking
    real_device_count = torch.accelerator.device_count()
    real_is_available = torch.accelerator.is_available()

    # Layer 1: DeviceRemapMode
    mode = DeviceRemapMode(fake_type=fake_type, real_type=real_type)
    mode.__enter__()
    config._emulate_device_remap_mode = mode

    # Layer 2: Mock accelerator identity
    # Use torch.device (not SimpleNamespace) because our migrated code passes
    # current_accelerator() to torch.device() constructor, which requires a
    # real torch.device object.
    fake_accel = torch.device(fake_type)
    torch.accelerator.current_accelerator = lambda: fake_accel
    torch.accelerator.device_count = lambda: real_device_count
    torch.accelerator.is_available = lambda: real_is_available
    torch.accelerator.current_device_index = lambda: 0

    # Layer 3: Patch is_accelerator_type to accept both types
    import compressed_tensors.offload.convert.helpers as _helpers

    config._emulate_orig_is_accelerator_type = _helpers.is_accelerator_type

    def patched_is_accelerator_type(device_type: str) -> bool:
        return device_type in (fake_type, real_type)

    _helpers.is_accelerator_type = patched_is_accelerator_type


def pytest_unconfigure(config):
    """Tear down device emulation — restore all patched objects."""
    mode = getattr(config, "_emulate_device_remap_mode", None)
    if mode is not None:
        mode.__exit__(None, None, None)

    orig_accel = getattr(config, "_emulate_orig_current_accelerator", None)
    if orig_accel is not None:
        torch.accelerator.current_accelerator = orig_accel
        torch.accelerator.device_count = config._emulate_orig_device_count
        torch.accelerator.is_available = config._emulate_orig_is_available
        torch.accelerator.current_device_index = (
            config._emulate_orig_current_device_index
        )

    orig_is_accel = getattr(config, "_emulate_orig_is_accelerator_type", None)
    if orig_is_accel is not None:
        import compressed_tensors.offload.convert.helpers as _helpers

        _helpers.is_accelerator_type = orig_is_accel
