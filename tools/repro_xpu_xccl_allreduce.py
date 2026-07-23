"""
Minimal reproduction for incorrect XPU/XCCL all_reduce MIN/MAX behavior.

Run with two visible XPUs, for example:

    .venv/bin/python -m torch.distributed.run --nproc_per_node 2 \
        tools/repro_xpu_xccl_allreduce.py

Expected on a correct backend:
    SUM -> [3.0, 4.0]
    MIN -> [1.0, 1.0]
    MAX -> [15.0, 20.0]

Observed on the affected XPU/XCCL setup:
    SUM -> [3.0, 4.0]
    MIN -> [3.0, 4.0]
    MAX -> [25.0, 30.0]
"""

import os
import sys

import torch
import torch.distributed as dist


def init_dist() -> torch.device:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    accel_type = torch.accelerator.current_accelerator().type
    device = torch.device(f"{accel_type}:{local_rank}")
    torch.accelerator.set_device_index(local_rank)

    if accel_type == "cuda":
        backend = "nccl"
    elif accel_type == "xpu":
        backend = "xccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    dist.barrier()
    return device


def main() -> int:
    device = init_dist()
    rank = dist.get_rank()
    backend = dist.get_backend()

    lhs = torch.tensor([1.0, 3.0], device=device) if rank == 0 else torch.tensor(
        [2.0, 1.0], device=device
    )
    rhs = torch.tensor([10.0, 20.0], device=device) if rank == 0 else torch.tensor(
        [15.0, 10.0], device=device
    )

    sum_result = lhs.clone()
    min_result = lhs.clone()
    max_result = rhs.clone()

    dist.all_reduce(sum_result, op=dist.ReduceOp.SUM)
    dist.all_reduce(min_result, op=dist.ReduceOp.MIN)
    work = dist.all_reduce(max_result, op=dist.ReduceOp.MAX, async_op=True)
    work.wait()

    expected_sum = torch.tensor([3.0, 4.0], device=device)
    expected_min = torch.tensor([1.0, 1.0], device=device)
    expected_max = torch.tensor([15.0, 20.0], device=device)

    local_ok = (
        torch.equal(sum_result, expected_sum)
        and torch.equal(min_result, expected_min)
        and torch.equal(max_result, expected_max)
    )

    print(
        {
            "rank": rank,
            "backend": backend,
            "device": str(device),
            "lhs_input": lhs.cpu().tolist(),
            "rhs_input": rhs.cpu().tolist(),
            "sum_result": sum_result.cpu().tolist(),
            "min_result": min_result.cpu().tolist(),
            "max_result": max_result.cpu().tolist(),
            "local_ok": local_ok,
        },
        flush=True,
    )

    ok_tensor = torch.tensor([1 if local_ok else 0], device=device)
    gathered = [torch.zeros_like(ok_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, ok_tensor)
    all_ok = all(t.item() == 1 for t in gathered)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if all_ok:
            print("XCCL all_reduce MIN/MAX behaved correctly", flush=True)
        else:
            print("Reproduced XCCL all_reduce MIN/MAX bug", flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
