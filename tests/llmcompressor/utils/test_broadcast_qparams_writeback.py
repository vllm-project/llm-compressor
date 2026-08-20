"""Regression tests for broadcast_qparams_and_cleanup writeback.

Validates that non-source ranks correctly persist broadcast results to
offload storage (CPUCache). Without the update_offload_parameter calls,
dist.broadcast modifies a temporary onloaded tensor but leaves the
underlying CPU storage stale — producing NaN weight_scale in DDP GPTQ
with the no-copy views path (issue #2949).
"""

import importlib
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as torch_dist

# Must use importlib to get the actual module object.
# `from .dist import *` in llmcompressor/utils/__init__.py re-exports
# `dist = torch.distributed`, so `import llmcompressor.utils.dist as m`
# resolves `m` to torch.distributed via attribute lookup — not the file module.
dist_module = importlib.import_module("llmcompressor.utils.dist")


def _make_module(**params):
    m = torch.nn.Linear(4, 4, bias=False)
    for name, val in params.items():
        setattr(m, name, val)
    return m


def _enter_dist_patches(stack, rank, world_size=2):
    """Stub out torch.distributed calls inside an ExitStack."""
    stack.enter_context(patch.object(torch_dist, "is_initialized", return_value=True))
    stack.enter_context(patch.object(torch_dist, "get_rank", return_value=rank))
    stack.enter_context(
        patch.object(torch_dist, "get_world_size", return_value=world_size)
    )
    stack.enter_context(patch.object(torch_dist, "broadcast", return_value=MagicMock()))


@pytest.mark.unit
def test_writeback_called_on_non_src_rank():
    """Non-src ranks must call update_offload_parameter for each broadcast param."""
    scale = torch.ones(4)
    zp = torch.zeros(4)
    mod = _make_module(weight_scale=scale, weight_zero_point=zp)
    module_to_rank = {mod: 0}

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        _enter_dist_patches(stack, rank=1)

        dist_module.broadcast_qparams_and_cleanup(
            [mod], module_to_rank, ["weight_scale", "weight_zero_point"]
        )

    assert mock_uop.call_count == 2
    mock_uop.assert_any_call(mod, "weight_scale", scale)
    mock_uop.assert_any_call(mod, "weight_zero_point", zp)


@pytest.mark.unit
def test_no_writeback_on_src_rank():
    """Source rank must not call update_offload_parameter (it owns the params)."""
    scale = torch.ones(4)
    mod = _make_module(weight_scale=scale)
    module_to_rank = {mod: 0}

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        _enter_dist_patches(stack, rank=0)

        dist_module.broadcast_qparams_and_cleanup(
            [mod], module_to_rank, ["weight_scale"]
        )

    mock_uop.assert_not_called()


@pytest.mark.unit
def test_writeback_skipped_for_missing_params():
    """Params absent on a module are silently skipped (no broadcast, no writeback)."""
    mod = _make_module()  # no weight_scale attribute
    module_to_rank = {mod: 0}

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        mock_broadcast = stack.enter_context(
            patch.object(torch_dist, "broadcast", return_value=MagicMock())
        )
        stack.enter_context(
            patch.object(torch_dist, "is_initialized", return_value=True)
        )
        stack.enter_context(patch.object(torch_dist, "get_rank", return_value=1))

        dist_module.broadcast_qparams_and_cleanup(
            [mod], module_to_rank, ["weight_scale"]
        )

    mock_broadcast.assert_not_called()
    mock_uop.assert_not_called()


@pytest.mark.unit
def test_writeback_covers_multiple_modules():
    """Writeback covers all non-src modules across multiple owner ranks."""
    scale_a = torch.ones(4)
    scale_b = torch.ones(4) * 2
    mod_a = _make_module(weight_scale=scale_a)
    mod_b = _make_module(weight_scale=scale_b)
    module_to_rank = {mod_a: 0, mod_b: 1}

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        _enter_dist_patches(stack, rank=2, world_size=3)

        dist_module.broadcast_qparams_and_cleanup(
            [mod_a, mod_b], module_to_rank, ["weight_scale"]
        )

    assert mock_uop.call_count == 2
    mock_uop.assert_any_call(mod_a, "weight_scale", scale_a)
    mock_uop.assert_any_call(mod_b, "weight_scale", scale_b)


@pytest.mark.unit
def test_copy_back_when_broadcast_param_is_distinct_storage():
    """When as_broadcastable returns a tensor with different storage (true copy),
    the broadcasted data must be copied back into param before writeback."""
    scale = torch.zeros(4)
    mod = _make_module(weight_scale=scale)
    module_to_rank = {mod: 0}

    # Simulate as_broadcastable returning a clone (different storage, different ptr)
    clone = scale.clone()
    clone.fill_(9.0)  # distinct values so we can verify the copy happened

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        stack.enter_context(
            patch.object(dist_module, "as_broadcastable", return_value=clone)
        )
        _enter_dist_patches(stack, rank=1)

        dist_module.broadcast_qparams_and_cleanup(
            [mod], module_to_rank, ["weight_scale"]
        )

    # param should have been updated with clone's values before writeback
    assert torch.all(scale == 9.0)
    mock_uop.assert_called_once_with(mod, "weight_scale", scale)


@pytest.mark.unit
def test_no_copy_when_broadcast_param_is_view_of_same_storage():
    """When as_broadcastable returns a view sharing storage with param (e.g. the
    FP8 uint8 reinterpret view), dist.broadcast already updates param in-place
    via the shared storage — param.copy_() must NOT be called."""
    scale = torch.zeros(4)
    mod = _make_module(weight_scale=scale)
    module_to_rank = {mod: 0}

    # Simulate as_broadcastable returning a view: different object, same data_ptr.
    view = scale.view(scale.numel())
    assert view.data_ptr() == scale.data_ptr()
    assert view is not scale

    # Simulate broadcast writing 7.0 into the shared storage.
    def _fake_broadcast(tensor, src, **kwargs):
        tensor.fill_(7.0)

    with ExitStack() as stack:
        mock_uop = stack.enter_context(
            patch.object(dist_module, "update_offload_parameter")
        )
        stack.enter_context(patch.object(dist_module, "_wait_for_comms"))
        stack.enter_context(
            patch.object(
                dist_module,
                "get_execution_device",
                return_value=torch.device("cuda", 0),
            )
        )
        stack.enter_context(
            patch.object(dist_module, "as_broadcastable", return_value=view)
        )
        stack.enter_context(
            patch.object(torch_dist, "broadcast", side_effect=_fake_broadcast)
        )
        stack.enter_context(
            patch.object(torch_dist, "is_initialized", return_value=True)
        )
        stack.enter_context(patch.object(torch_dist, "get_rank", return_value=1))
        stack.enter_context(patch.object(torch_dist, "get_world_size", return_value=2))

        dist_module.broadcast_qparams_and_cleanup(
            [mod], module_to_rank, ["weight_scale"]
        )

    # Broadcast wrote into the shared storage — param reflects the update without copy_.
    assert torch.all(scale == 7.0)
    # update_offload_parameter is still called with param (not view).
    mock_uop.assert_called_once_with(mod, "weight_scale", scale)
