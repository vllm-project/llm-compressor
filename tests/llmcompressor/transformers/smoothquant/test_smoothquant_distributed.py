"""
Unit tests for distributed SmoothQuantModifier.

These are mock-based tests that verify the all_reduce call contract
without requiring GPU hardware.

Multi-GPU integration tests have been moved to:
    tests/llmcompressor/transformers/compression/test_compression_ddp.py

Run unit tests:
    pytest tests/.../test_smoothquant_distributed.py -m unit -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed

# ---------------------------------------------------------------------------
# Unit tests — mock-based, no GPU required
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reduce_activation_scales_noop_single_gpu():
    """_reduce_activation_scales must be a no-op when is_distributed() is False."""
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {
        "layer0": SmoothQuantScale(
            min_channel_vals=torch.tensor([-1.0, -2.0]),
            max_channel_vals=torch.tensor([1.0, 2.0]),
        )
    }

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=False,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist") as mock_dist,
    ):
        modifier._reduce_activation_scales()
        mock_dist.all_reduce.assert_not_called()


@pytest.mark.unit
def test_reduce_activation_scales_2n_calls_for_n_layers():
    """For N layers, exactly 2*N async all_reduce calls must be batched."""
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    n = 4
    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {
        f"layer{i}": SmoothQuantScale(
            min_channel_vals=torch.zeros(16),
            max_channel_vals=torch.ones(16),
        )
        for i in range(n)
    }

    collected = []

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist") as mock_dist,
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.wait_for_comms",
            side_effect=lambda h: collected.extend(h),
        ),
    ):
        mock_dist.all_reduce.return_value = MagicMock()
        mock_dist.ReduceOp.MIN = torch.distributed.ReduceOp.MIN
        mock_dist.ReduceOp.MAX = torch.distributed.ReduceOp.MAX

        modifier._reduce_activation_scales()

        assert mock_dist.all_reduce.call_count == 2 * n
        assert len(collected) == 2 * n


@pytest.mark.unit
def test_apply_smoothing_calls_reduce_only_when_distributed():
    """
    _reduce_activation_scales() must be called inside _apply_smoothing()
    only when is_distributed() is True, and skipped otherwise.
    """
    from llmcompressor.modifiers.transform.smoothquant.base import SmoothQuantModifier

    modifier = SmoothQuantModifier()
    modifier.scales_ = {}
    modifier.resolved_mappings_ = []

    # Distributed case: reduce should be called
    order = []
    with patch.object(
        modifier,
        "_reduce_activation_scales",
        side_effect=lambda: order.append("reduce"),
    ):
        with patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ):
            modifier._apply_smoothing(model=MagicMock())
    assert order == ["reduce"]

    # Single-GPU case: reduce should NOT be called
    order_single = []
    with patch.object(
        modifier,
        "_reduce_activation_scales",
        side_effect=lambda: order_single.append("reduce"),
    ):
        with patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=False,
        ):
            modifier._apply_smoothing(model=MagicMock())
    assert order_single == []
