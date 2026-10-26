"""
Unit tests for the REAP saliency tracker's reduction behavior.

Distributed (data-parallel) REAP runs reduce *every* tracked layer at every
sequential epoch end, so layers which have not seen tokens yet keep their
trackers and must keep collecting on their accumulation device in later
epochs. ``reduce_saliency_stats`` therefore runs the reduction on float32
copies on the reduction device (the accelerator, whose gpu-backed process
groups only support float32) and must leave the persistent accumulators
untouched.

The collectives are faked with single-member semantics: ``reduce`` delivers
each rank's own values to the destination and ``all_reduce`` is the identity,
which is exactly what a world size of one contributes.
"""

import pytest
import torch

from llmcompressor.modifiers.pruning.reap import utils as reap_utils
from llmcompressor.modifiers.pruning.reap.utils import REAPSaliencyTracker


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
    monkeypatch.setattr(reap_utils, "is_distributed", lambda: True)
    monkeypatch.setattr(reap_utils, "is_source_process", lambda: source)
    monkeypatch.setattr(reap_utils, "get_source_rank", lambda: 0)
    monkeypatch.setattr(reap_utils, "dist", _FakeDist(device))
    monkeypatch.setattr(reap_utils, "wait_for_comms", lambda pending: None)


def _accumulate(tracker: REAPSaliencyTracker, device: torch.device):
    topk_indices = torch.tensor([[0, 1], [0, 2]], device=device)
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], device=device)
    expert_norms_dict = {
        0: torch.tensor([3.0, 1.0], device=device),
        1: torch.tensor([2.0], device=device),
        2: torch.tensor([4.0], device=device),
    }
    tracker.update(topk_indices, topk_weights, expert_norms_dict)


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

    tracker = REAPSaliencyTracker(num_experts=4)
    tracker.reduce_saliency_stats(device)
    assert tracker.count is None
    assert tracker.sum_saliency is None
    assert tracker.total_count == 0.0

    # data arrives in a later epoch and accumulates from scratch
    _accumulate(tracker, device)
    assert tracker.sum_saliency is not None
    assert tracker.sum_saliency.tolist() == pytest.approx([2.3, 0.8, 2.0, 0.0])
    assert tracker.count.tolist() == [2.0, 1.0, 1.0, 0.0]
    assert tracker.total_count == 4.0


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_publishes_to_source_rank_once_tokens_seen(monkeypatch, device):
    """Once the layer has seen tokens, the source rank takes over the reduced
    statistics (in float32 on the reduction device) for the consumers which
    follow: report and retained-expert selection."""
    _fake_distributed(monkeypatch, device, source=True)

    tracker = REAPSaliencyTracker(num_experts=4)
    _accumulate(tracker, device)

    tracker.reduce_saliency_stats(device)
    assert tracker.total_count == 4.0
    assert tracker.sum_saliency.device == device
    assert tracker.sum_saliency.dtype == torch.float32
    assert tracker.sum_saliency.tolist() == pytest.approx([2.3, 0.8, 2.0, 0.0])
    assert tracker.count.tolist() == [2.0, 1.0, 1.0, 0.0]


@pytest.mark.unit
@pytest.mark.parametrize("device", _reduction_devices())
def test_reduce_stats_skips_publish_on_non_source_ranks(monkeypatch, device):
    _fake_distributed(monkeypatch, device, source=False)

    tracker = REAPSaliencyTracker(num_experts=4)
    _accumulate(tracker, device)

    tracker.reduce_saliency_stats(device)
    # non-source ranks keep accumulating locally for the next reduction
    assert tracker.sum_saliency.tolist() == pytest.approx([2.3, 0.8, 2.0, 0.0])

    _accumulate(tracker, device)
    assert tracker.sum_saliency.tolist() == pytest.approx([4.6, 1.6, 4.0, 0.0])
    assert tracker.total_count == 8.0


@pytest.mark.unit
def test_reduce_stats_is_a_no_op_when_not_distributed():
    tracker = REAPSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))
    before = tracker.sum_saliency.clone()

    tracker.reduce_saliency_stats(torch.device("cpu"))
    assert (
        tracker.sum_saliency is before
        or tracker.sum_saliency.tolist() == before.tolist()
    )
    assert tracker.total_count == 4.0


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
    tracker = REAPSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))
    assert tracker.sum_saliency.device.type == "cpu"

    tracker.reduce_saliency_stats(device)
    assert tracker.sum_saliency.device.type == "cpu"
    _accumulate(tracker, torch.device("cpu"))
    assert tracker.sum_saliency.tolist() == pytest.approx([4.6, 1.6, 4.0, 0.0])

    # source rank: takes over the reduced copies on the reduction device
    _fake_distributed(monkeypatch, device, source=True)
    tracker = REAPSaliencyTracker(num_experts=4)
    _accumulate(tracker, torch.device("cpu"))

    tracker.reduce_saliency_stats(device)
    assert tracker.sum_saliency.device == device
    assert tracker.sum_saliency.tolist() == pytest.approx([2.3, 0.8, 2.0, 0.0])
