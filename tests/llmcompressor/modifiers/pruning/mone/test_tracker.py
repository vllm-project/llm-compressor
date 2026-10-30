"""
Unit tests for the MoNe modifier's per-layer statistics reduction behavior.

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

from llmcompressor.modifiers.pruning import base as pruning_base
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier


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


def _modifier_with_layer(device: torch.device) -> MoNePruningModifier:
    modifier = MoNePruningModifier(sparsity=0.5)
    modifier._init_layer_stats("layer", 4, 4, device)
    return modifier


def _accumulate(modifier: MoNePruningModifier, device: torch.device):
    topk_indices = torch.tensor([[0, 1], [0, 2]], device=device)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.5]], device=device)
    modifier._update_routing("layer", topk_indices, topk_weights)
    modifier._update_outputs(
        "layer", topk_indices, {0: torch.ones(2, 4, device=device)}
    )


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
        modifier._sum_output["layer"],
        modifier._sum_sq_output["layer"],
        modifier._output_counts["layer"],
        modifier._sum_gates["layer"],
    ):
        assert tensor.device == device
        assert tensor.dtype == torch.float32
        assert torch.all(tensor == 0)
    assert modifier._sum_output["layer"].shape == (4, 4)
    assert modifier._token_counts["layer"] == 0


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_zero_token_layer_contributes_zeros(monkeypatch, device):
    """A layer without tokens yet contributes its zero-initialized
    accumulators to the reduction and keeps collecting afterwards."""
    _fake_distributed(monkeypatch, device)

    modifier = _modifier_with_layer(device)
    modifier._reduce_layer_stats("layer")
    assert modifier._token_counts["layer"] == 0
    assert modifier._sum_gates["layer"].tolist() == [0.0, 0.0, 0.0, 0.0]

    # data arrives in a later epoch and accumulates from the zeros
    _accumulate(modifier, device)
    assert modifier._sum_gates["layer"].tolist() == [0.625, 0.25, 0.5, 0.0]
    assert modifier._token_counts["layer"] == 2


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_reduces_accumulators_in_place(monkeypatch, device):
    """``dist.reduce`` only writes its result at the destination, so the
    accumulators are reduced in place (never copied or moved) and hold the
    global sums for the consumers which follow: report, retained-expert
    selection, novices."""
    _fake_distributed(monkeypatch, device)

    modifier = _modifier_with_layer(device)
    _accumulate(modifier, device)
    sum_gates = modifier._sum_gates["layer"]

    modifier._reduce_layer_stats("layer")
    assert modifier._token_counts["layer"] == 2
    # same tensor, still on the accumulation device in float32 (single-member
    # semantics leave the values unchanged)
    assert modifier._sum_gates["layer"] is sum_gates
    assert modifier._sum_gates["layer"].device == device
    assert modifier._sum_gates["layer"].dtype == torch.float32
    assert modifier._sum_gates["layer"].tolist() == [0.625, 0.25, 0.5, 0.0]
    assert modifier._sum_output["layer"][0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert modifier._output_counts["layer"].tolist() == [2.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_reduce_stats_is_a_no_op_when_not_distributed():
    modifier = _modifier_with_layer(torch.device("cpu"))
    _accumulate(modifier, torch.device("cpu"))
    sum_gates = modifier._sum_gates["layer"]

    modifier._reduce_layer_stats("layer")
    assert modifier._sum_gates["layer"] is sum_gates
    assert modifier._token_counts["layer"] == 2
