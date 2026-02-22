"""
Run with: torchrun --nproc_per_node=2 -m pytest <this_file> -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from llmcompressor.utils.distributed import (
    all_reduce_max,
    all_reduce_min,
    get_rank,
    get_world_size,
    is_distributed,
)
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
    if not is_distributed():
        pytest.skip("Requires torchrun --nproc_per_node=2")


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_all_reduce_min_max():
    _skip_if_not_distributed()
    rank = get_rank()

    mins = (
        torch.tensor([1.0, 3.0], device="cuda")
        if rank == 0
        else torch.tensor([2.0, 1.0], device="cuda")
    )
    maxs = (
        torch.tensor([10.0, 20.0], device="cuda")
        if rank == 0
        else torch.tensor([15.0, 10.0], device="cuda")
    )

    assert torch.equal(all_reduce_min(mins), torch.tensor([1.0, 1.0], device="cuda"))
    assert torch.equal(all_reduce_max(maxs), torch.tensor([15.0, 20.0], device="cuda"))


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_synced_qparams_are_identical_across_ranks():
    _skip_if_not_distributed()
    rank = get_rank()

    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.utils import calculate_qparams

    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")

    local_min = (
        torch.tensor([-2.0], device="cuda")
        if rank == 0
        else torch.tensor([-5.0], device="cuda")
    )
    local_max = (
        torch.tensor([3.0], device="cuda")
        if rank == 0
        else torch.tensor([1.0], device="cuda")
    )

    global_min = all_reduce_min(local_min.clone())
    global_max = all_reduce_max(local_max.clone())

    scale, _ = calculate_qparams(
        min_vals=global_min,
        max_vals=global_max,
        quantization_args=args,
    )

    gathered = [torch.zeros_like(scale) for _ in range(get_world_size())]
    dist.all_gather(gathered, scale)
    assert torch.equal(gathered[0], gathered[1])
