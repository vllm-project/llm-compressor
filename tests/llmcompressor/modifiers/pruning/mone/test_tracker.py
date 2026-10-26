"""
Unit tests for the MoNe saliency tracker's reduction behavior.

Distributed (data-parallel) MoNe runs reduce *every* tracked layer at every
sequential epoch end, so layers which have not seen tokens yet keep their
trackers and must keep collecting on their accumulation device in later
epochs. ``reduce_stats`` therefore runs the reduction on float32 copies on
the reduction device (the accelerator, whose gpu-backed process groups only
support float32) and must leave the persistent accumulators untouched.

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
        assert tensor.device == self.device
        assert tensor.dtype == torch.float32, "gpu-backed groups reduce float32"
        return _FakeHandle()

    def all_reduce(self, tensor, op=None, async_op=False, group=None):
        assert tensor.device == self.device
        assert tensor.dtype == torch.float32, "gpu-backed groups reduce float32"
        return _FakeHandle()


def _fake_distributed(monkeypatch, device: torch.device, source: bool):
    monkeypatch.setattr(mone_utils, "is_distributed", lambda: True)
    monkeypatch.setattr(mone_utils, "is_source_process", lambda: source)
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
def test_reduce_stats_leaves_zero_token_accumulators_untouched(monkeypatch, device):
    """A layer without tokens yet must not gain accumulators from a reduce;
    it keeps lazily allocating on the device of whatever data arrives in a
    later epoch."""
    _fake_distributed(monkeypatch, device, source=True)

    tracker = MoNeSaliencyTracker(num_experts=4)
    tracker.reduce_stats(hidden_dim=4, device=device)
    assert tracker.sum_gate is None
    assert tracker.sum_output is None
    assert tracker.token_count == 0

    # data arrives in a later epoch and accumulates from scratch
    _accumulate(tracker, device)
    assert tracker.sum_gate is not None
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]
    assert tracker.token_count == 2


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_publishes_to_source_rank_once_tokens_seen(monkeypatch, device):
    """Once the layer has seen tokens, the source rank takes over the reduced
    statistics (in float32 on the reduction device) for the consumers which
    follow: report, retained-expert selection, novices."""
    _fake_distributed(monkeypatch, device, source=True)

    tracker = MoNeSaliencyTracker(num_experts=4)
    _accumulate(tracker, device)

    tracker.reduce_stats(hidden_dim=4, device=device)
    assert tracker.token_count == 2
    assert tracker.sum_gate.device == device
    assert tracker.sum_gate.dtype == torch.float32
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]
    assert tracker.sum_output[0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert tracker.output_count.tolist() == [2.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_skips_publish_on_non_source_ranks(monkeypatch, device):
    _fake_distributed(monkeypatch, device, source=False)

    tracker = MoNeSaliencyTracker(num_experts=4)
    _accumulate(tracker, device)

    tracker.reduce_stats(hidden_dim=4, device=device)
    # non-source ranks keep accumulating locally for the next reduction
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]

    _accumulate(tracker, device)
    assert tracker.sum_gate.tolist() == [1.25, 0.5, 1.0, 0.0]


@pytest.mark.unit
def test_reduce_stats_is_a_no_op_when_not_distributed():
    tracker = MoNeSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))
    before = tracker.sum_gate.clone()

    tracker.reduce_stats(hidden_dim=4, device=torch.device("cpu"))
    assert tracker.sum_gate is before or tracker.sum_gate.tolist() == before.tolist()
    assert tracker.token_count == 2


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="requires an accelerator"
)
def test_reduce_stats_keeps_cpu_accumulators_when_reducing_on_gpu(monkeypatch):
    """Accumulators collected on the cpu must stay there (later batches keep
    accumulating on the cpu) even when the reduction runs on the gpu; only
    the source rank takes over the reduced copies, which are consumed
    immediately after."""
    accel_type = torch.accelerator.current_accelerator().type
    device = torch.device(accel_type, 0)

    # non-source rank: the persistent accumulators stay on the cpu
    _fake_distributed(monkeypatch, device, source=False)
    tracker = MoNeSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))
    assert tracker.sum_gate.device.type == "cpu"

    tracker.reduce_stats(hidden_dim=4, device=device)
    assert tracker.sum_gate.device.type == "cpu"
    _accumulate(tracker, torch.device("cpu"))
    assert tracker.sum_gate.tolist() == [1.25, 0.5, 1.0, 0.0]

    # source rank: takes over the reduced copies on the reduction device
    _fake_distributed(monkeypatch, device, source=True)
    tracker = MoNeSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))

    tracker.reduce_stats(hidden_dim=4, device=device)
    assert tracker.sum_gate.device == device
    assert tracker.sum_gate.tolist() == [0.625, 0.25, 0.5, 0.0]
