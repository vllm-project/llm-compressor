from typing import Callable, Hashable, TypeVar

import torch.distributed as dist

T = TypeVar("T", bound=Hashable)


def greedy_bin_packing(
    items: list[T],
    num_bins: int,
    item_weight_fn: Callable[[T], float] = lambda x: 1,
) -> tuple[list[T], list[list[T]], dict[T, int]]:
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
    items.sort(key=item_weight_fn, reverse=True)
    bin_to_items: list[list[T]] = [[] for _ in range(num_bins)]
    item_to_bin: dict[T, int] = dict()
    bin_weights: list[float] = [0 for _ in range(num_bins)]
    for item in items:
        target_bin = bin_weights.index(min(bin_weights))
        bin_to_items[target_bin].append(item)
        item_to_bin[item] = target_bin
        bin_weights[target_bin] += item_weight_fn(item)
    return items, bin_to_items, item_to_bin


def wait_for_comms(pending_comms: list[dist.Work]) -> None:
    """Block until all pending async distributed operations complete.

    Calls ``wait()`` on each work handle, then clears the list in-place
    so it can be reused for the next batch of operations.

    :param pending_comms: mutable list of async communication handles
        (returned by ``dist.reduce``, ``dist.broadcast``, etc. with
        ``async_op=True``). The list is cleared after all operations
        have completed.
    """
    for comm in list(pending_comms):
        comm.wait()
    pending_comms.clear()
