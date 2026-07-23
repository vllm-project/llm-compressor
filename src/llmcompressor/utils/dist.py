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
from compressed_tensors.offload import get_execution_device
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

    :param module_list: all modules across all ranks
    :param module_to_rank: mapping from module to the rank that computed its qparams
    :param qparam_names: attribute names to broadcast (e.g. weight_scale, weight)
    :param skip_cpu: if True, skip broadcasting for CPU-offloaded modules
    """
    pending_comms = []
    for module in module_list:
        should_broadcast = not skip_cpu or (
            get_execution_device(module) != torch.device("cpu")
        )
        if should_broadcast:
            for name in qparam_names:
                if (param := getattr(module, name, None)) is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(param),
                            src=module_to_rank[module],
                            async_op=True,
                        )
                    )

        obs = getattr(module, "weight_observer", None)
        if obs is not None and obs.has_statistics:
            obs.delete_statistics(check_fused=True)

    _wait_for_comms(pending_comms)
