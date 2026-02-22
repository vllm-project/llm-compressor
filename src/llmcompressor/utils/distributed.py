"""
Distributed utilities for multi-GPU (DDP) calibration and optimization.
All functions are no-ops when torch.distributed is not initialized.
"""

from typing import Dict, List, Tuple

import torch
from compressed_tensors.utils import get_execution_device, update_offload_parameter
from loguru import logger
from torch import distributed as dist
from torch.nn import Module

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "partition_modules_by_weight_size",
    "build_module_to_rank_map",
    "broadcast_module_parameter",
    "all_reduce_min",
    "all_reduce_max",
]


def is_distributed() -> bool:
    """Return True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current rank, or 0 if not distributed."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the world size, or 1 if not distributed."""
    return dist.get_world_size() if is_distributed() else 1


def _compute_rank_assignments(
    named_modules: List[Tuple[str, Module]],
    world_size: int,
) -> List[List[Tuple[str, Module]]]:
    """
    Assign modules to ranks using greedy bin-packing by weight size.

    Sorts modules by weight.numel() descending and assigns each to the
    rank with the smallest current load. This produces a deterministic
    assignment that all ranks agree on.

    :param named_modules: list of (name, module) pairs
    :param world_size: number of ranks
    :return: list of lists, where index i contains modules assigned to rank i
    """
    weighted = [
        (name, mod, mod.weight.numel() if hasattr(mod, "weight") else 0)
        for name, mod in named_modules
    ]
    weighted.sort(key=lambda x: x[2], reverse=True)

    rank_loads = [0] * world_size
    rank_assignments: List[List[Tuple[str, Module]]] = [[] for _ in range(world_size)]

    for name, mod, size in weighted:
        target_rank = rank_loads.index(min(rank_loads))
        rank_assignments[target_rank].append((name, mod))
        rank_loads[target_rank] += size

    return rank_assignments


def partition_modules_by_weight_size(
    named_modules: List[Tuple[str, Module]],
) -> List[Tuple[str, Module]]:
    """
    Partition modules across ranks proportional to weight.numel().
    Returns only the modules assigned to the current rank.

    Uses greedy bin-packing for load balancing.
    If not distributed, returns all modules.

    :param named_modules: list of (name, module) pairs
    :return: subset of named_modules assigned to this rank
    """
    if not is_distributed():
        return named_modules

    world_size = get_world_size()
    rank = get_rank()

    assignments = _compute_rank_assignments(named_modules, world_size)

    logger.debug(
        "Rank {} assigned {} of {} modules for weight calibration",
        rank,
        len(assignments[rank]),
        len(named_modules),
    )

    return assignments[rank]


def build_module_to_rank_map(
    named_modules: List[Tuple[str, Module]],
) -> Dict[Module, int]:
    """
    Build a map from module to the rank that owns it.

    Uses the same deterministic assignment as partition_modules_by_weight_size,
    so all ranks produce the same map without communication.

    :param named_modules: list of (name, module) pairs
    :return: dict mapping each module to its assigned rank
    """
    if not is_distributed():
        return {mod: 0 for _, mod in named_modules}

    world_size = get_world_size()
    assignments = _compute_rank_assignments(named_modules, world_size)

    module_to_rank: Dict[Module, int] = {}
    for rank_idx, rank_modules in enumerate(assignments):
        for _, mod in rank_modules:
            module_to_rank[mod] = rank_idx

    return module_to_rank


def broadcast_module_parameter(
    module: Module,
    param_name: str,
    src_rank: int,
):
    """
    Broadcast a module parameter from src_rank to all ranks. Handles
    CPU-offloaded parameters. No-op if not distributed.

    :param module: module containing the parameter
    :param param_name: name of the parameter to broadcast
    :param src_rank: rank that has the computed value
    """
    if not is_distributed():
        return

    param = getattr(module, param_name, None)
    if param is None:
        return

    device = get_execution_device(module)
    tensor = param.data.to(device)
    dist.broadcast(tensor, src=src_rank)
    update_offload_parameter(module, param_name, tensor)


def all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor with MIN op across all ranks. No-op if not distributed.

    :param tensor: tensor to reduce
    :return: reduced tensor
    """
    if not is_distributed():
        return tensor

    device = tensor.device
    if device.type == "cpu":
        tensor = tensor.cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor.to(device)

    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor with MAX op across all ranks. No-op if not distributed.

    :param tensor: tensor to reduce
    :return: reduced tensor
    """
    if not is_distributed():
        return tensor

    device = tensor.device
    if device.type == "cpu":
        tensor = tensor.cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.to(device)

    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor
