# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transform phase status, plus the retained per-parameter writeback experiment."""

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_distributed
from compressed_tensors.offload import OffloadCache, update_offload_parameter
from compressed_tensors.offload.cache import DistributedCPUCache, DistributedDiskCache


def finish_transform_phase(error: Exception | None, stage: str):
    """One CPU status reduction at a transform phase/mapping boundary.

    The Ascend driver supplies cpu:gloo,npu:hccl. A failure invalidates the model;
    no rollback is attempted. This also waits for every rank's local work.
    """
    if is_distributed():
        failed = torch.tensor(int(error is not None))
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(f"{stage} failed; discard model") from error
    elif error is not None:
        raise error


@torch.no_grad()
def update_transform_parameter(module: torch.nn.Module, name: str, data: torch.Tensor):
    """Commit an already computed transform after every rank finishes its old read.

    Experimental primitive only; QuaRot/FlexSmooth use phase/mapping coordination
    instead of this per-parameter protocol.

    All WORLD ranks must call with the same parameter order and compatible CT
    caches. The transform inputs/outputs must agree across replicas; this helper
    coordinates writes, not distributed transform computation. Only the CT source
    rank updates shared CPU/disk backing. Every resident copy is refreshed from
    that backing before returning. Private device/dict caches update locally.

    Supports existing floating parameters/buffers with unchanged shape/dtype.
    Does not broadcast weight tensors, change checkpoint indexes, or initialize
    distributed state. On a write/refresh failure all ranks raise; callers must
    mark the model failed, since there is no rollback of partially written data.
    """
    if not dist.is_initialized():
        update_offload_parameter(module, name, data)
        return

    # Snapshot even a view of shared CPU storage before any rank can mutate it.
    snapshot = data.detach().clone()
    cache, current = None, None
    with OffloadCache.disable_onloading():
        for mapping in (module._parameters, module._buffers):
            if name in mapping:
                current = mapping[name]
                cache = mapping if isinstance(mapping, OffloadCache) else None
                break
    kind = (
        2
        if isinstance(cache, DistributedDiskCache)
        else 1
        if isinstance(cache, DistributedCPUCache)
        else 0
    )
    valid = (
        current is not None
        and current.shape == snapshot.shape
        and current.dtype == snapshot.dtype
        and snapshot.is_floating_point()
        and snapshot.layout == torch.strided
        # is_shared alone does not prove all ranks refer to the same storage.
        # Only CT's distributed cache construction establishes that contract.
        and not (kind == 0 and current.device.type == "cpu" and current.is_shared())
        and bool(torch.isfinite(snapshot).all())
    )
    info = torch.tensor([int(valid), kind], device=snapshot.device, dtype=torch.int64)
    peers = [torch.empty_like(info) for _ in range(dist.get_world_size())]
    # This agreement is also the read-before-write barrier.
    dist.all_gather(peers, info)
    if any(peer.tolist() != [1, kind] for peer in peers):
        raise ValueError("Invalid or inconsistent collective transform writeback")

    def agree_failure(error, stage):
        failed = torch.tensor([int(error is not None)], device=snapshot.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(
                f"Collective transform {stage} failed; discard model"
            ) from error

    error = None
    try:
        if kind == 0 or dist.get_rank() == get_source_rank():
            update_offload_parameter(module, name, snapshot)
    except Exception as caught:
        error = caught
    # Wait for the sole shared writer, or every private writer, to finish.
    agree_failure(error, "write")

    error = None
    try:
        if kind and cache is not None and dist.get_rank() != get_source_rank():
            resident = cache.keep_onloaded_values.get(current)
            if resident is not None and resident is not current:
                # Copy into the same object: external resident references stay fresh.
                # Do not use non-source results as authoritative shared data.
                resident.copy_(cache.onload(current))
    except Exception as caught:
        error = caught
    # Prevent the next transform from racing any rank's refresh.
    agree_failure(error, "refresh")
