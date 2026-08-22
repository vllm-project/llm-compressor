from collections.abc import Sequence
from typing import Hashable, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.distributed import (
    greedy_bin_packing as _greedy_bin_packing,
)
from compressed_tensors.distributed import (
    wait_for_comms as _wait_for_comms,
)
from compressed_tensors.offload import get_execution_device, update_offload_parameter
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.utils.helpers import deprecated

T = TypeVar("T", bound=Hashable)


@deprecated("compressed_tensors.distributed.assign::greedy_bin_packing")
def greedy_bin_packing(*args, **kwargs) -> tuple[list[T], list[list[T]], dict[T, int]]:
    """Distribute items across bins using a greedy bin-packing heuristic.

    Items are sorted by weight in descending order, then each item is
    assigned to the bin with the smallest current total weight. This
    approximates an even distribution of weight across bins.

    :param items: items to distribute. Sorted in-place by descending weight.
    :param num_bins: number of bins to distribute items across.
    :param item_weight_fn: callable that returns the weight of an item.
        Defaults to uniform weight of 1.
    :return: a 3-tuple of:
        - items: the input list, now sorted by descending weight.
        - bin_to_items: list of length ``num_bins`` where each element is
          the list of items assigned to that bin.
        - item_to_bin: mapping from each item to its assigned bin index.
    """
    return _greedy_bin_packing(*args, **kwargs)


@deprecated("compressed_tensors.distributed.utils::wait_for_comms")
def wait_for_comms(*args, **kwargs) -> None:
    """Block until all pending async distributed operations complete.

    Calls ``wait()`` on each work handle, then clears the list in-place
    so it can be reused for the next batch of operations.

    :param pending_comms: mutable list of async communication handles
        (returned by ``dist.reduce``, ``dist.broadcast``, etc. with
        ``async_op=True``). The list is cleared after all operations
        have completed.
    """
    return _wait_for_comms(*args, **kwargs)


def broadcast_qparams_and_cleanup(
    module_list: list[torch.nn.Module],
    module_to_rank: dict[torch.nn.Module, int],
    qparam_names: Sequence[str],
    skip_cpu: bool = True,
) -> None:
    """Broadcast quantization params from owning rank and clean up observer stats.

    For CPU-offloaded modules (e.g. those using CPUCache), ``dist.broadcast``
    modifies a temporary onloaded tensor in-place rather than the underlying
    offload storage. The explicit ``update_offload_parameter`` calls after
    ``_wait_for_comms`` write the broadcast result back to the actual storage
    on non-source ranks.

    :param module_list: all modules across all ranks
    :param module_to_rank: mapping from module to the rank that computed its qparams
    :param qparam_names: attribute names to broadcast (e.g. weight_scale, weight)
    :param skip_cpu: if True, skip broadcasting for CPU-offloaded modules
    """
    is_dist_initialized = dist.is_initialized()
    rank = dist.get_rank() if is_dist_initialized else 0

    pending_comms = []
    writeback_items: list[tuple[torch.nn.Module, str, torch.Tensor, torch.Tensor]] = []

    for module in module_list:
        should_broadcast = is_dist_initialized and (
            not skip_cpu or (get_execution_device(module) != torch.device("cpu"))
        )
        if should_broadcast:
            src = module_to_rank[module]
            for name in qparam_names:
                if (param := getattr(module, name, None)) is not None:
                    broadcast_param = as_broadcastable(param)
                    pending_comms.append(
                        dist.broadcast(
                            broadcast_param,
                            src=src,
                            async_op=True,
                        )
                    )
                    if rank != src:
                        writeback_items.append((module, name, param, broadcast_param))

        obs = getattr(module, "weight_observer", None)
        if obs is not None and obs.has_statistics:
            obs.delete_statistics(check_fused=True)

    _wait_for_comms(pending_comms)

    for module, name, param, broadcast_param in writeback_items:
        if broadcast_param is not param and (
            broadcast_param.data_ptr() != param.data_ptr()
        ):
            param.copy_(broadcast_param.view_as(param))
        update_offload_parameter(module, name, param)
