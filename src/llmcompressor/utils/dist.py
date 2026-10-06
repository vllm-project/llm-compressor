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
from compressed_tensors.offload.cache import DeviceCache
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.utils.helpers import deprecated
from loguru import logger

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


def _is_device_offload(module: torch.nn.Module) -> bool:
    """Whether a module's weights reside on a different GPU than they execute on.

    GPU-offloaded modules (``DeviceCache``, including its distributed variant) keep
    their canonical weights in ``offloaded_values`` on the offload device and move
    them to the onload device only during the forward pass. When the two devices
    differ, the offloaded copy must be kept in sync explicitly, since an in-place
    update to the onloaded copy alone is lost once the module is offloaded again.
    """
    cache = module._parameters
    return (
        isinstance(cache, DeviceCache) and cache.onload_device != cache.offload_device
    )


def broadcast_qparams_and_cleanup(
    module_list: list[torch.nn.Module],
    module_to_rank: dict[torch.nn.Module, int],
    qparam_names: Sequence[str],
    skip_cpu: bool = True,
) -> None:
    """Broadcast quantization params from owning rank and clean up observer stats.

    In addition to updating the onloaded copy on every rank, the offloaded copy of
    GPU-offloaded modules (``DeviceCache`` with ``onload_device != offload_device``)
    is updated on every non-owning rank: the broadcast-received value is copied
    locally into the offloaded copy via ``OffloadCache.update_offload``. The copy is
    deferred until ``_wait_for_comms`` completes, because the write to the broadcast
    target tensor is not observable before the collective work is waited on. The
    owning rank skips the local copy: it already wrote the compressed value into
    the offloaded copy during write-back. CPU-offloaded and single-GPU behavior is
    unchanged.

    :param module_list: all modules across all ranks
    :param module_to_rank: mapping from module to the rank that computed its qparams
    :param qparam_names: attribute names to broadcast (e.g. weight_scale, weight)
    :param skip_cpu: if True, skip broadcasting for CPU-offloaded modules
    """
    pending_comms = []
    # offloaded copies to refresh after all broadcasts complete: the broadcast write
    # to ``param`` is not visible until the collective work is waited on, so the
    # local copy into the offloaded value must be deferred to that point
    offload_refresh: list[tuple[DeviceCache, torch.Tensor, torch.Tensor]] = []
    for module in module_list:
        should_broadcast = not skip_cpu or (
            get_execution_device(module) != torch.device("cpu")
        )
        is_device_offload = _is_device_offload(module)
        # the owning rank updated the offloaded copy during qparam write-back
        is_src_rank = dist.is_initialized() and (
            dist.get_rank() == module_to_rank[module]
        )
        if should_broadcast:
            cache = module._parameters if is_device_offload else None
            for name in qparam_names:
                if (param := getattr(module, name, None)) is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(param),
                            src=module_to_rank[module],
                            async_op=True,
                        )
                    )
                    # keep the GPU-offloaded copy in sync once the broadcast lands
                    if cache is not None and not is_src_rank:
                        offloaded = cache.offloaded_values.get(name)
                        if offloaded is not None and torch.is_same_size(
                            offloaded, param
                        ):
                            offload_refresh.append((cache, offloaded, param))
                        elif offloaded is not None:
                            logger.warning(
                                "Skipping GPU-offload sync for {}: offloaded shape "
                                "{} != onloaded shape {}",
                                name,
                                tuple(offloaded.shape),
                                tuple(param.shape),
                            )

        obs = getattr(module, "weight_observer", None)
        if obs is not None and obs.has_statistics:
            obs.delete_statistics(check_fused=True)

    _wait_for_comms(pending_comms)
    for cache, offloaded, param in offload_refresh:
        cache.update_offload(offloaded, param)
