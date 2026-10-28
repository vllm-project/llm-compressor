"""
Unit tests for the MoNe saliency tracker's reduction behavior.

Distributed (data-parallel) MoNe runs reduce *every* tracked layer at every
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

from llmcompressor.modifiers.pruning.mone import utils as mone_utils
from llmcompressor.modifiers.pruning.mone.utils import MoNeSaliencyTracker


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
    monkeypatch.setattr(mone_utils, "is_distributed", lambda: True)
    monkeypatch.setattr(mone_utils, "get_source_rank", lambda: 0)
    monkeypatch.setattr(mone_utils, "dist", _FakeDist(device))
    monkeypatch.setattr(mone_utils, "wait_for_comms", lambda pending: None)


def _accumulate(tracker: MoNeSaliencyTracker, device: torch.device):
    topk_indices = torch.tensor([[0, 1], [0, 2]], device=device)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.5]], device=device)
    tracker.update_routing(topk_indices, topk_weights)
    tracker.update_outputs(topk_indices, {0: torch.ones(2, 4, device=device)})


def _reduction_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        devices.append(torch.device(accel_type, 0))
    return devices


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_accumulators_are_initialized_on_the_device(monkeypatch, device):
    """The accumulators are zero-initialized on ``device`` at construction, in
    float32, so every rank always holds tensors to contribute to a reduction."""
    _fake_distributed(monkeypatch, device)

    tracker = MoNeSaliencyTracker(num_experts=4, hidden_dim=4, device=device)
    assert tracker.device == device
    assert tracker.hidden_dim == 4
    for tensor in (
        tracker.sum_output,
        tracker.sum_sq_output,
        tracker.output_count,
        tracker.sum_gate,
    ):
        assert tensor.device == device
        assert tensor.dtype == torch.float32
        assert torch.all(tensor == 0)
    assert tracker.sum_output.shape == (4, 4)


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_zero_token_layer_contributes_zeros(monkeypatch, device):
    """A layer without tokens yet contributes its zero-initialized
    accumulators to the reduction and keeps collecting afterwards."""
    _fake_distributed(monkeypatch, device)

    tracker = MoNeSaliencyTracker(num_experts=4, hidden_dim=4, device=device)
    tracker.reduce_stats()
    assert tracker.token_count == 0
    assert tracker.sum_gate.tolist() == [0.0, 0.0, 0.0, 0.0]

    # data arrives in a later epoch and accumulates from the zeros
    _accumulate(tracker, device)
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]
    assert tracker.token_count == 2


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_reduces_accumulators_in_place(monkeypatch, device):
    """``dist.reduce`` only writes its result at the destination, so the
    accumulators are reduced in place (never copied or moved) and hold the
    global sums for the consumers which follow: report, retained-expert
    selection, novices."""
    _fake_distributed(monkeypatch, device)

    tracker = MoNeSaliencyTracker(num_experts=4, hidden_dim=4, device=device)
    _accumulate(tracker, device)
    sum_gate = tracker.sum_gate

    tracker.reduce_stats()
    assert tracker.token_count == 2
    # same tensor, still on the accumulation device in float32 (single-member
    # semantics leave the values unchanged)
    assert tracker.sum_gate is sum_gate
    assert tracker.sum_gate.device == device
    assert tracker.sum_gate.dtype == torch.float32
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]
    assert tracker.sum_output[0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert tracker.output_count.tolist() == [2.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_reduce_stats_is_a_no_op_when_not_distributed():
    tracker = MoNeSaliencyTracker(
        num_experts=4, hidden_dim=4, device=torch.device("cpu")
    )
    _accumulate(tracker, torch.device("cpu"))
    sum_gate = tracker.sum_gate

    tracker.reduce_stats()
    assert tracker.sum_gate is sum_gate
    assert tracker.token_count == 2
