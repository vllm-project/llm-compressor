"""
Unit tests for distributed broadcast of quantization params, focusing on GPU
offloading (``DeviceCache`` with distinct onload/offload devices).

All tests run on CPU: device movement is stubbed and ``dist.broadcast`` is
patched so no CUDA or process group is required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.offload import OffloadCache
from compressed_tensors.offload.cache import DeviceCache

from llmcompressor.utils.dist import (
    _is_device_offload,
    broadcast_qparams_and_cleanup,
)

_DIST_MODULE = "llmcompressor.utils.dist"
_QPARAMS = ["weight", "weight_scale"]
# value a fake broadcast "receives"; used to trace propagation into offloaded copies
_SENTINEL = 7.0


class _FakeDeviceCache(DeviceCache):
    """DeviceCache stand-in that never moves tensors (CPU-runnable)."""

    def __init__(self, onload_device, offload_device, values):
        super().__init__(onload_device, offload_device)
        self.offloaded_values = values

    def onload(self, offloaded):
        # mirror DeviceCache semantics: onloaded copy is distinct from offloaded
        return offloaded.clone() if offloaded is not None else None

    def offload(self, tensor):
        return tensor


class _FakeOffloadCache(OffloadCache):
    """Non-DeviceCache offload cache (CPU-offload semantics), no device movement."""

    def __init__(self, onload_device, offload_device, values):
        super().__init__(onload_device, offload_device)
        self.offloaded_values = values

    def stage(self, offloaded, pin_memory=False):
        return offloaded

    def onload(self, offloaded):
        return offloaded.clone() if offloaded is not None else None

    def offload(self, tensor):
        return tensor

    def update_offload(self, offloaded, data):
        if offloaded is not None and data is not None:
            offloaded.copy_(data)


def _make_module(cache):
    module = torch.nn.Module()
    module._parameters = cache
    return module


def _run_broadcast(modules, module_to_rank, qparam_names=_QPARAMS, **kwargs):
    """Run broadcast_qparams_and_cleanup, recording dist.broadcast calls."""
    if not isinstance(modules, list):
        modules = [modules]
    calls = []

    def fake_broadcast(tensor, src, async_op=False, **kwargs):
        # simulate a broadcast receive: overwrite the destination tensor
        tensor.fill_(_SENTINEL)
        calls.append((tensor, src, async_op))
        return MagicMock()

    with (
        patch("torch.distributed.broadcast", side_effect=fake_broadcast),
        patch(_DIST_MODULE + "._wait_for_comms"),
    ):
        broadcast_qparams_and_cleanup(modules, module_to_rank, qparam_names, **kwargs)
    return calls


@pytest.mark.unit
def test_is_device_offload_detects_gpu_offload():
    cache = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        {"weight": torch.randn(4, 4)},
    )
    assert _is_device_offload(_make_module(cache))

    same_device = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:0"),
        {"weight": torch.randn(4, 4)},
    )
    assert not _is_device_offload(_make_module(same_device))


@pytest.mark.unit
def test_is_device_offload_false_for_cpu_cache_and_plain_modules():
    cpu_cache = _FakeOffloadCache(
        torch.device("cuda:0"),
        torch.device("cpu"),
        {"weight": torch.randn(4, 4)},
    )
    assert not _is_device_offload(_make_module(cpu_cache))

    plain = torch.nn.Linear(4, 4, bias=False)
    assert not _is_device_offload(plain)


@pytest.mark.unit
def test_gpu_offload_syncs_offloaded_copies_after_broadcast():
    """DeviceCache with distinct devices: onload broadcast then local offload sync."""
    cache = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        {"weight": torch.randn(4, 4), "weight_scale": torch.randn(1)},
    )
    module = _make_module(cache)
    rank = 1

    calls = _run_broadcast(module, {module: rank})

    # only the onloaded copies are broadcast (2 qparams)
    assert len(calls) == 2
    assert all(src == rank for _, src, _ in calls)

    # the broadcast-received values were propagated into the offloaded copies
    for name in _QPARAMS:
        assert cache.offloaded_values[name].eq(_SENTINEL).all()


@pytest.mark.unit
def test_owning_rank_skips_offload_copy_sync():
    """The rank that computed the qparams already updated its offloaded copy."""
    cache = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        {"weight": torch.randn(4, 4), "weight_scale": torch.randn(1)},
    )
    module = _make_module(cache)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        calls = _run_broadcast(module, {module: 0})

    # owner rank still broadcasts (onload copies) but skips the local sync
    assert len(calls) == 2
    for name in _QPARAMS:
        assert not cache.offloaded_values[name].eq(_SENTINEL).any()


@pytest.mark.unit
def test_gpu_offload_same_device_leaves_offloaded_copies_alone():
    """DeviceCache with onload == offload (single-GPU resident): onload only."""
    cache = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:0"),
        {"weight": torch.randn(4, 4), "weight_scale": torch.randn(1)},
    )
    module = _make_module(cache)

    calls = _run_broadcast(module, {module: 0})

    assert len(calls) == 2  # onload copies only
    for name in _QPARAMS:
        assert not cache.offloaded_values[name].eq(_SENTINEL).any()


@pytest.mark.unit
def test_cpu_offload_fallback_leaves_offloaded_copies_alone():
    """CPU-offload cache: behavior unchanged (onload copies only)."""
    cache = _FakeOffloadCache(
        torch.device("cuda:0"),
        torch.device("cpu"),
        {"weight": torch.randn(4, 4), "weight_scale": torch.randn(1)},
    )
    module = _make_module(cache)

    calls = _run_broadcast(module, {module: 0})

    assert len(calls) == 2
    for name in _QPARAMS:
        assert not cache.offloaded_values[name].eq(_SENTINEL).any()


@pytest.mark.unit
def test_skip_cpu_execution_no_broadcast():
    """skip_cpu=True with CPU execution device: nothing is broadcast."""
    module = torch.nn.Linear(4, 4, bias=False)  # params on cpu

    calls = _run_broadcast(module, {module: 0})

    assert calls == []


@pytest.mark.unit
def test_offloaded_none_entries_are_skipped():
    """None entries in offloaded_values (e.g. bias=None) must not be touched."""
    cache = _FakeDeviceCache(
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        {"weight": torch.randn(4, 4), "weight_scale": None},
    )
    module = _make_module(cache)

    calls = _run_broadcast(module, {module: 0})

    # only weight is broadcast; weight_scale (None) is skipped on both copies
    assert len(calls) == 1
    for tensor, _, _ in calls:
        assert tensor is not None
    assert cache.offloaded_values["weight"].eq(_SENTINEL).all()
    assert cache.offloaded_values["weight_scale"] is None


@pytest.mark.unit
def test_observer_stats_cleaned_up_even_when_skipped():
    module = torch.nn.Linear(4, 4, bias=False)
    obs = MagicMock()
    obs.has_statistics = True
    module.weight_observer = obs

    calls = _run_broadcast(module, {module: 0})

    assert calls == []  # cpu execution -> skipped
    obs.delete_statistics.assert_called_once_with(check_fused=True)


@pytest.mark.unit
def test_empty_module_list_safe_without_dist_init():
    """broadcast_qparams_and_cleanup is a no-op for an empty list (non-distributed)."""
    calls = _run_broadcast([], {})
    assert calls == []


@pytest.mark.unit
def test_invalid_staging_device_raises():
    """Invalid offload targets surface a clean error at cache selection."""
    with pytest.raises(NotImplementedError):
        OffloadCache.cls_from_device("meta")


@pytest.mark.unit
def test_gptq_non_distributed_skips_broadcast():
    """GPTQModifier.compress_modules avoids broadcast entirely when not distributed."""
    from llmcompressor.modifiers.gptq import GPTQModifier

    modifier = GPTQModifier(block_size=16)
    with (
        patch("llmcompressor.modifiers.gptq.base.is_distributed", return_value=False),
        # pydantic model instances block setattr of non-field attributes, so the
        # method is patched at class level instead of on the instance
        patch(
            "llmcompressor.modifiers.gptq.base.GPTQModifier.compress_module_list"
        ) as mock_cml,
        patch("llmcompressor.utils.dist.broadcast_qparams_and_cleanup") as mock_bc,
    ):
        modifier.compress_modules()

    mock_cml.assert_called_once_with([])
    mock_bc.assert_not_called()
