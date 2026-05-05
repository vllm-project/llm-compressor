"""
Multi-GPU tests for quantization with distributed data parallel.

These tests use the @torchrun decorator which automatically spawns
torchrun when run with regular pytest.
"""

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.distributed import wait_for_comms
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.min_max import StaticMinMaxObserver
from tests.testing_utils import requires_gpu, torchrun


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_observer_synchronize_reduces_min_max():
    init_dist()
    rank = dist.get_rank()

    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = StaticMinMaxObserver(base_name="input", args=args)

    # each rank has different local statistics
    observer.min_vals = (
        torch.tensor([1.0, 3.0], device="cuda")
        if rank == 0
        else torch.tensor([2.0, 1.0], device="cuda")
    )
    observer.max_vals = (
        torch.tensor([10.0, 20.0], device="cuda")
        if rank == 0
        else torch.tensor([15.0, 10.0], device="cuda")
    )

    comms = observer.sync_activation_stats()
    wait_for_comms(comms)

    # after sync, min should be element-wise minimum, max element-wise maximum
    assert torch.equal(observer.min_vals, torch.tensor([1.0, 1.0], device="cuda"))
    assert torch.equal(observer.max_vals, torch.tensor([15.0, 20.0], device="cuda"))


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_synced_qparams_are_identical_across_ranks():
    init_dist()
    rank = dist.get_rank()

    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = StaticMinMaxObserver(base_name="input", args=args)

    observer.min_vals = (
        torch.tensor([-2.0], device="cuda")
        if rank == 0
        else torch.tensor([-5.0], device="cuda")
    )
    observer.max_vals = (
        torch.tensor([3.0], device="cuda")
        if rank == 0
        else torch.tensor([1.0], device="cuda")
    )

    comms = observer.sync_activation_stats()
    wait_for_comms(comms)

    qparams = observer.get_qparams()
    scale = qparams["scale"]

    gathered = [torch.zeros_like(scale) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, scale)
    assert torch.equal(gathered[0], gathered[1])
