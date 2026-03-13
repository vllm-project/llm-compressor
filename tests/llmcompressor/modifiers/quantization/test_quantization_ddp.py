"""
Run with: torchrun --nproc_per_node=2 -m pytest <this_file> -v
"""

import os

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.min_max import StaticMinMaxObserver
from llmcompressor.utils.dist import wait_for_comms
from tests.testing_utils import requires_gpu

# initialize process group when running under torchrun
if (
    os.environ.get("RANK") is not None
    and torch.cuda.is_available()
    and not dist.is_initialized()
):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def _skip_if_not_distributed():
    if not (dist.is_available() and dist.is_initialized()):
        pytest.skip("Requires torchrun --nproc_per_node=2")


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_observer_synchronize_reduces_min_max():
    _skip_if_not_distributed()
    rank = dist.get_rank()

    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = StaticMinMaxObserver(base_name="input", args=args)

    # each rank has different local statistics
    observer.past_min_vals = (
        torch.tensor([1.0, 3.0], device="cuda")
        if rank == 0
        else torch.tensor([2.0, 1.0], device="cuda")
    )
    observer.past_max_vals = (
        torch.tensor([10.0, 20.0], device="cuda")
        if rank == 0
        else torch.tensor([15.0, 10.0], device="cuda")
    )

    comms = observer.synchronize()
    wait_for_comms(comms)

    # after sync, min should be element-wise minimum, max element-wise maximum
    assert torch.equal(observer.past_min_vals, torch.tensor([1.0, 1.0], device="cuda"))
    assert torch.equal(
        observer.past_max_vals, torch.tensor([15.0, 20.0], device="cuda")
    )


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_synced_qparams_are_identical_across_ranks():
    _skip_if_not_distributed()
    rank = dist.get_rank()

    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = StaticMinMaxObserver(base_name="input", args=args)

    observer.past_min_vals = (
        torch.tensor([-2.0], device="cuda")
        if rank == 0
        else torch.tensor([-5.0], device="cuda")
    )
    observer.past_max_vals = (
        torch.tensor([3.0], device="cuda")
        if rank == 0
        else torch.tensor([1.0], device="cuda")
    )

    comms = observer.synchronize()
    wait_for_comms(comms)

    result = observer.recompute_qparams()
    assert result is not None
    scale, _ = result

    gathered = [torch.zeros_like(scale) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, scale)
    assert torch.equal(gathered[0], gathered[1])
