"""Root-cause proof: CPUCache.__getattr__ returns a temporary CUDA tensor.

In-place modifications (e.g. from dist.broadcast) are silently discarded
without an explicit update_offload_parameter call. This is the underlying
mechanism behind the broadcast_qparams_and_cleanup writeback bug.

Requires 1 GPU. No distributed setup, no model weights, no calibration data.

Setup that reproduces the bug path:
  - offload_module(..., onload_device='cuda:0', offload_device='cpu') → CPUCache
  - register_parameter('weight_scale', ...) → stored in CPUCache._parameters
  - getattr(mod, 'weight_scale') → CPUCache onloads to GPU → returns NEW CUDA tensor
  - dist.broadcast modifies that temporary in-place
  - temporary is discarded → CPU storage retains original (stale) value

Note: setattr / __dict__ attributes do NOT trigger this behavior.
      CPU/CPU offload does NOT trigger this behavior (no device transfer).
"""

import pytest
import torch
from compressed_tensors.offload import offload_module, update_offload_parameter


def _make_offloaded_module_with_scale(scale_value: float) -> torch.nn.Module:
    """CPUCache-offloaded module with weight_scale as a registered parameter."""
    mod = torch.nn.Linear(4, 4, bias=False)
    offload_module(mod, onload_device="cuda:0", offload_device="cpu")
    # register_parameter mirrors what quantization lifecycle does for weight_scale
    mod.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.full((4,), scale_value), requires_grad=False),
    )
    return mod


@pytest.mark.unit
def test_cpucache_getattr_returns_new_temporary_each_call():
    """Each getattr call on a CPUCache-offloaded registered parameter returns
    a freshly allocated CUDA tensor — not the same object or same storage.

    This is the key property that makes dist.broadcast's in-place write
    ineffective: it writes to a temporary that is immediately discarded.
    """
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA")

    mod = _make_offloaded_module_with_scale(2.0)

    first = getattr(mod, "weight_scale")
    second = getattr(mod, "weight_scale")

    assert first.device.type == "cuda", "CPUCache should onload to CUDA"
    assert first is not second, "each getattr must return a distinct tensor object"
    assert (
        first.data_ptr() != second.data_ptr()
    ), "each getattr allocates a new CUDA buffer — data_ptrs must differ"


@pytest.mark.unit
def test_inplace_modification_lost_without_writeback():
    """Simulates exactly what dist.broadcast does: modifies the temporary
    CUDA tensor in-place. Without update_offload_parameter the modification
    is silently discarded — the CPU offload storage keeps the original value.

    This is the bug: non-source ranks end up with stale weight_scale.
    """
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA")

    mod = _make_offloaded_module_with_scale(2.0)

    # Step 1: retrieve temporary CUDA tensor (mirrors getattr in
    # broadcast_qparams_and_cleanup)
    tmp = getattr(mod, "weight_scale")
    assert tmp.device.type == "cuda"

    # Step 2: simulate dist.broadcast writing into the temporary in-place
    tmp.fill_(99.0)
    assert torch.all(tmp == 99.0)  # modification visible on the temporary

    # Step 3: re-read — CPUCache allocates a fresh onload; original CPU value survives
    after = getattr(mod, "weight_scale")
    assert torch.all(after == 2.0), (
        f"Expected original 2.0 to survive (writeback not called), got {after.cpu()} — "
        "CPUCache storage was unexpectedly updated without update_offload_parameter"
    )


@pytest.mark.unit
def test_update_offload_parameter_persists_modification():
    """update_offload_parameter correctly flushes the modified CUDA tensor
    back into CPUCache CPU storage — this is the mechanism the fix relies on.

    After calling it, subsequent getattr returns the new value.
    """
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA")

    mod = _make_offloaded_module_with_scale(2.0)

    tmp = getattr(mod, "weight_scale")
    tmp.fill_(99.0)

    # Explicit writeback — what the fix adds after _wait_for_comms
    update_offload_parameter(mod, "weight_scale", tmp)

    after = getattr(mod, "weight_scale")
    assert torch.all(
        after == 99.0
    ), f"Expected 99.0 after update_offload_parameter, got {after.cpu()}"
