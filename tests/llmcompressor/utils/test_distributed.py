from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.nn import Linear, Module

from llmcompressor.utils.distributed import (
    _compute_rank_assignments,
    all_reduce_max,
    all_reduce_min,
    get_rank,
    get_world_size,
    is_distributed,
    partition_modules_by_weight_size,
)


class TinyModel(Module):
    def __init__(self):
        super().__init__()
        self.small = Linear(16, 32)
        self.medium = Linear(64, 128)
        self.large = Linear(128, 256)


def _named_modules(model):
    return [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, Linear)
    ]


@pytest.mark.unit
def test_not_distributed_by_default():
    assert not is_distributed()
    assert get_rank() == 0
    assert get_world_size() == 1


@pytest.mark.unit
def test_partition_returns_all_when_not_distributed():
    model = TinyModel()
    named = _named_modules(model)
    assert len(partition_modules_by_weight_size(named)) == len(named)


@pytest.mark.unit
def test_assignments_are_balanced_and_complete():
    model = TinyModel()
    named = _named_modules(model)
    assignments = _compute_rank_assignments(named, world_size=2)

    all_assigned = [mod for rank_mods in assignments for _, mod in rank_mods]
    assert len(all_assigned) == len(named)

    total_per_rank = [
        sum(m.weight.numel() for _, m in rank_mods) for rank_mods in assignments
    ]
    assert all(t > 0 for t in total_per_rank)
    assert max(total_per_rank) / sum(total_per_rank) < 0.85


@pytest.mark.unit
def test_all_reduce_noop_when_not_distributed():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(all_reduce_min(t), t)
    assert torch.equal(all_reduce_max(t), t)


@pytest.mark.unit
@patch("llmcompressor.utils.distributed.dist")
def test_all_reduce_calls_correct_ops(mock_dist):
    mock_dist.is_available.return_value = True
    mock_dist.is_initialized.return_value = True
    mock_dist.ReduceOp.MIN = "MIN"
    mock_dist.ReduceOp.MAX = "MAX"

    t = MagicMock(wraps=torch.tensor([1.0]))
    t.device = torch.device("cuda:0")

    all_reduce_min(t)
    mock_dist.all_reduce.assert_called_with(t, op="MIN")

    mock_dist.all_reduce.reset_mock()
    all_reduce_max(t)
    mock_dist.all_reduce.assert_called_with(t, op="MAX")
