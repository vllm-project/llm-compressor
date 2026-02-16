import torch
import torch.distributed as dist
import contextlib

def greedy_bin_packing(items: list, num_bins: int, item_weight_fn=lambda x: 1):
    """
    1) Sort items by weight
    2) for each item, assign it to the bin with the least total weight
    """
    items.sort(key=item_weight_fn, reverse=True)
    bin_to_items = [[] for _ in range(num_bins)]
    item_to_bin = dict()
    bin_weights = [0 for _ in range(num_bins)]
    for item in items:
        target_bin = bin_weights.index(min(bin_weights))
        bin_to_items[target_bin].append(item)
        item_to_bin[item] = target_bin
        bin_weights[target_bin] += item_weight_fn(item)
    return items, bin_to_items, item_to_bin

def wait_for_comms(pending_comms):
    for comm in list(pending_comms):
        comm.wait()
    pending_comms.clear()