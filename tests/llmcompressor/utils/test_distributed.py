from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.min_max import (
    MemorylessMinMaxObserver,
    MinMaxObserver,
    StaticMinMaxObserver,
)


def _make_observer(cls, **kwargs):
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    return cls(base_name="input", args=args, **kwargs)


@pytest.mark.unit
def test_memoryless_synchronize_returns_empty():
    observer = _make_observer(MemorylessMinMaxObserver)
    assert observer.synchronize() == []


@pytest.mark.unit
def test_memoryless_recompute_returns_none():
    observer = _make_observer(MemorylessMinMaxObserver)
    assert observer.recompute_qparams() is None
    assert observer.recompute_global_scale() is None


@pytest.mark.unit
def test_static_synchronize_returns_empty_before_observation():
    observer = _make_observer(StaticMinMaxObserver)
    assert observer.synchronize() == []


@pytest.mark.unit
@patch("llmcompressor.observers.base.dist")
def test_static_synchronize_issues_all_reduce(mock_dist):
    mock_dist.ReduceOp.MIN = "MIN"
    mock_dist.ReduceOp.MAX = "MAX"
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(StaticMinMaxObserver)
    observer.past_min_vals = torch.tensor([-1.0])
    observer.past_max_vals = torch.tensor([1.0])

    comms = observer.synchronize()
    assert len(comms) == 2
    assert mock_dist.all_reduce.call_count == 2

    # verify correct ops
    calls = mock_dist.all_reduce.call_args_list
    assert calls[0].kwargs["op"] == "MIN"
    assert calls[1].kwargs["op"] == "MAX"


@pytest.mark.unit
@patch("llmcompressor.observers.base.dist")
def test_static_synchronize_with_global_state(mock_dist):
    mock_dist.ReduceOp.MIN = "MIN"
    mock_dist.ReduceOp.MAX = "MAX"
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(StaticMinMaxObserver)
    observer.past_min_vals = torch.tensor([-1.0])
    observer.past_max_vals = torch.tensor([1.0])
    observer.past_global_min_vals = torch.tensor([-2.0])
    observer.past_global_max_vals = torch.tensor([2.0])

    comms = observer.synchronize()
    assert len(comms) == 4
    assert mock_dist.all_reduce.call_count == 4


@pytest.mark.unit
@patch("llmcompressor.observers.moving_base.dist")
def test_moving_avg_synchronize_issues_all_reduce(mock_dist):
    mock_dist.ReduceOp.SUM = "SUM"
    mock_dist.get_world_size.return_value = 2
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(MinMaxObserver)
    observer.past_min_vals = torch.tensor([-1.0])
    observer.past_max_vals = torch.tensor([1.0])

    comms = observer.synchronize()
    assert len(comms) == 2


@pytest.mark.unit
def test_recompute_qparams_from_accumulated_state():
    observer = _make_observer(StaticMinMaxObserver)
    observer.past_min_vals = torch.tensor([-5.0])
    observer.past_max_vals = torch.tensor([5.0])

    result = observer.recompute_qparams()
    assert result is not None
    scale, zero_point = result
    assert scale.numel() > 0
    assert zero_point.numel() > 0


@pytest.mark.unit
def test_recompute_qparams_returns_none_without_state():
    observer = _make_observer(StaticMinMaxObserver)
    assert observer.recompute_qparams() is None


@pytest.mark.unit
def test_recompute_global_scale_returns_none_without_state():
    observer = _make_observer(StaticMinMaxObserver)
    assert observer.recompute_global_scale() is None


@pytest.mark.unit
def test_recompute_global_scale_from_accumulated_state():
    observer = _make_observer(StaticMinMaxObserver)
    observer.past_global_min_vals = torch.tensor([-10.0])
    observer.past_global_max_vals = torch.tensor([10.0])

    result = observer.recompute_global_scale()
    assert result is not None
    assert result.numel() > 0
