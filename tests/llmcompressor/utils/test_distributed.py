from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
from torch import distributed as dist

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
    assert observer.synchronize_observer() == []


@pytest.mark.unit
def test_memoryless_get_qparams_raises_without_observation():
    observer = _make_observer(MemorylessMinMaxObserver)
    with pytest.raises(RuntimeError, match="No statistics available"):
        observer.get_qparams()


@pytest.mark.unit
def test_static_synchronize_returns_empty_before_observation():
    observer = _make_observer(StaticMinMaxObserver)
    assert observer.synchronize_observer() == []


@pytest.mark.unit
@patch("llmcompressor.observers.base.dist")
def test_static_synchronize_issues_all_reduce(mock_dist):
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(StaticMinMaxObserver)
    observer.min_vals = torch.tensor([-1.0])
    observer.max_vals = torch.tensor([1.0])

    comms = observer.synchronize_observer()
    assert len(comms) == 2
    assert mock_dist.all_reduce.call_count == 2

    # verify correct ops
    calls = mock_dist.all_reduce.call_args_list
    assert calls[0].kwargs["op"] == dist.ReduceOp.MIN
    assert calls[1].kwargs["op"] == dist.ReduceOp.MAX


@pytest.mark.unit
@patch("llmcompressor.observers.base.dist")
def test_static_synchronize_with_global_state(mock_dist):
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(StaticMinMaxObserver)
    observer.min_vals = torch.tensor([-1.0])
    observer.max_vals = torch.tensor([1.0])

    comms = observer.synchronize_observer()
    assert len(comms) == 2
    assert mock_dist.all_reduce.call_count == 2


@pytest.mark.unit
@patch("llmcompressor.observers.base.dist")
def test_moving_avg_synchronize_issues_all_reduce(mock_dist):
    mock_dist.all_reduce.return_value = MagicMock()

    observer = _make_observer(MinMaxObserver)
    observer.min_vals = torch.tensor([-1.0])
    observer.max_vals = torch.tensor([1.0])

    comms = observer.synchronize_observer()
    assert len(comms) == 2

    # verify AVG ops used
    calls = mock_dist.all_reduce.call_args_list
    assert calls[0].kwargs["op"] == dist.ReduceOp.AVG
    assert calls[1].kwargs["op"] == dist.ReduceOp.AVG


@pytest.mark.unit
def test_get_qparams_from_accumulated_state():
    observer = _make_observer(StaticMinMaxObserver)
    observer.min_vals = torch.tensor([-5.0])
    observer.max_vals = torch.tensor([5.0])

    qparams = observer.get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )
    assert scale.numel() > 0
    assert zero_point.numel() > 0
    # global_scale is None for non-TENSOR_GROUP strategies
    assert global_scale is None


@pytest.mark.unit
def test_get_qparams_raises_without_state():
    observer = _make_observer(StaticMinMaxObserver)
    with pytest.raises(RuntimeError, match="No statistics"):
        observer.get_qparams()


@pytest.mark.unit
def test_get_qparams_with_tensor_group_strategy():
    args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, strategy="tensor_group", group_size=4
    )
    # Need a module for TENSOR_GROUP strategy
    module = torch.nn.Linear(8, 4)
    observer = StaticMinMaxObserver(base_name="weight", args=args, module=module)
    observer.min_vals = torch.tensor([[-5.0]])
    observer.max_vals = torch.tensor([[5.0]])

    qparams = observer.get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )
    assert scale.numel() > 0
    assert zero_point.numel() > 0
    assert global_scale is not None
    assert global_scale.numel() > 0
