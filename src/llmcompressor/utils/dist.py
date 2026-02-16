import torch
import torch.distributed as dist
import contextlib

def greedy_bin_packing(items: list, num_bins: int, item_weight_fn=lambda x: 1):
    """
    1) Sort items by weight
    2) for each item, assign it to the bin with the least total weight
    """
    items.sort(key=item_weight_fn, reverse=True)
    bin_to_item = [[] for _ in range(num_bins)]
    item_to_bin = dict()
    bin_weights = [0 for _ in range(num_bins)]
    for item in items:
        target_bin = bin_weights.index(min(bin_weights))
        bin_to_item[target_bin].append(item)
        item_to_bin[item] = target_bin
        bin_weights[target_bin] += item_weight_fn(item)
    return items, bin_to_item, item_to_bin


def _dist_comms_impl(
    keys,
    key_to_rank,
    get_data_fn=lambda key: None,
    comm_fn=lambda data, target_rank: None,
    store_data_fn=lambda key, data: None,
    should_store_data_fn= lambda target_rank: False,
    context_fn = None,
):  
    if context_fn is None:
        context_fn = lambda x: contextlib.nullcontext()
    
    pending_comms = []
    for key in keys:
        target_rank = key_to_rank[key]
        with context_fn(key):
            data = get_data_fn(key)
            comm = comm_fn(data, target_rank)
            pending_comms.extend([comm] if not isinstance(comm, list) else comm)
            if should_store_data_fn(target_rank):
                wait_for_comms(pending_comms)
                store_data_fn(key, data)
    wait_for_comms(pending_comms)

def wait_for_comms(pending_comms):
    for comm in list(pending_comms):
        comm.wait()
    pending_comms.clear()