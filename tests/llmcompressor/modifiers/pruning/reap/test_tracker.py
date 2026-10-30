"""
Unit tests for the REAP modifier's per-layer saliency reduction behavior.

Distributed (data-parallel) REAP runs reduce *every* tracked layer at every
sequential epoch end over the default process group, in float32 on the device
the statistics were accumulated on. The accumulators are zero-initialized at
construction and reduced in place, so a layer which has not seen tokens yet
contributes zeros and keeps collecting afterwards.

The collectives are faked with single-member semantics: ``reduce`` delivers
each rank's own values to the destination and ``all_reduce`` is the identity,
which is exactly what a world size of one contributes.
"""

import pytest
import torch

from llmcompressor.modifiers.pruning import base as pruning_base
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier


class _FakeHandle:
    def wait(self):
        pass


class _FakeDist:
    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, device: torch.device):
        self.device = device

    def reduce(self, tensor, dst=None, op=None, async_op=False, group=None):
        assert group is None, "reductions run on the default process group"
        assert tensor.device == self.device
        assert tensor.dtype == torch.float32, "gpu-backed groups reduce float32"
        return _FakeHandle()

    def all_reduce(self, tensor, op=None, async_op=False, group=None):
        assert group is None, "reductions run on the default process group"
        assert tensor.device == self.device
        assert tensor.dtype == torch.float32, "gpu-backed groups reduce float32"
        return _FakeHandle()


def _fake_distributed(monkeypatch, device: torch.device):
    monkeypatch.setattr(pruning_base, "is_distributed", lambda: True)
    monkeypatch.setattr(pruning_base, "get_source_rank", lambda: 0)
    monkeypatch.setattr(pruning_base, "dist", _FakeDist(device))
    monkeypatch.setattr(pruning_base, "wait_for_comms", lambda pending: None)


def _modifier_with_layer(device: torch.device) -> REAPPruningModifier:
    modifier = REAPPruningModifier(sparsity=0.5)
    modifier._layer_devices["layer"] = device
    modifier._init_layer_stats("layer", 4, 2, device)
    return modifier


def _accumulate(modifier: REAPPruningModifier, device: torch.device):
    topk_indices = torch.tensor([[0, 1], [0, 2]], device=device)
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], device=device)
    expert_norms_dict = {
        0: torch.tensor([3.0, 1.0], device=device),
        1: torch.tensor([2.0], device=device),
        2: torch.tensor([4.0], device=device),
    }
    modifier._update_stats("layer", topk_indices, topk_weights, expert_norms_dict)


def _reduction_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        devices.append(torch.device(accel_type, 0))
    return devices


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_accumulators_are_initialized_on_the_device(monkeypatch, device):
    """The accumulators are zero-initialized on ``device`` at calibration
    start, in float32, so every rank always holds tensors to contribute to a
    reduction."""
    _fake_distributed(monkeypatch, device)

    modifier = _modifier_with_layer(device)
    for tensor in (
        modifier._layer_stats["layer"]["sum_saliency"],
        modifier._layer_stats["layer"]["counts"],
    ):
        assert tensor.device == device
        assert tensor.dtype == torch.float32
        assert torch.all(tensor == 0)
    assert modifier._layer_counts["layer"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_zero_token_layer_contributes_zeros(monkeypatch, device):
    """A layer without tokens yet contributes its zero-initialized
    accumulators to the reduction and keeps collecting afterwards."""
    _fake_distributed(monkeypatch, device)

    modifier = _modifier_with_layer(device)
    modifier._reduce_layer_stats("layer")
    assert modifier._layer_counts["layer"] == 0.0
    assert modifier._layer_stats["layer"]["counts"].tolist() == [0.0, 0.0, 0.0, 0.0]

    # data arrives in a later epoch and accumulates from the zeros
    _accumulate(modifier, device)
    assert modifier._layer_stats["layer"]["sum_saliency"].tolist() == pytest.approx(
        [2.3, 0.8, 2.0, 0.0]
    )
    assert modifier._layer_stats["layer"]["counts"].tolist() == [2.0, 1.0, 1.0, 0.0]
    assert modifier._layer_counts["layer"] == 4.0


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_reduces_accumulators_in_place(monkeypatch, device):
    """``dist.reduce`` only writes its result at the destination, so the
    accumulators are reduced in place (never copied or moved) and hold the
    global sums for the consumers which follow: report and retained-expert
    selection."""
    _fake_distributed(monkeypatch, device)

    modifier = _modifier_with_layer(device)
    _accumulate(modifier, device)
    sum_saliency = modifier._layer_stats["layer"]["sum_saliency"]

    modifier._reduce_layer_stats("layer")
    assert modifier._layer_counts["layer"] == 4.0
    # same tensor, still on the accumulation device in float32 (single-member
    # semantics leave the values unchanged)
    assert modifier._layer_stats["layer"]["sum_saliency"] is sum_saliency
    assert modifier._layer_stats["layer"]["sum_saliency"].device == device
    assert modifier._layer_stats["layer"]["sum_saliency"].dtype == torch.float32
    assert modifier._layer_stats["layer"]["sum_saliency"].tolist() == pytest.approx(
        [2.3, 0.8, 2.0, 0.0]
    )
    assert modifier._layer_stats["layer"]["counts"].tolist() == [2.0, 1.0, 1.0, 0.0]


@pytest.mark.unit
def test_reduce_stats_is_a_no_op_when_not_distributed():
    modifier = _modifier_with_layer(torch.device("cpu"))
    _accumulate(modifier, torch.device("cpu"))
    sum_saliency = modifier._layer_stats["layer"]["sum_saliency"]

    modifier._reduce_layer_stats("layer")
    assert modifier._layer_stats["layer"]["sum_saliency"] is sum_saliency
    assert modifier._layer_counts["layer"] == 4.0
