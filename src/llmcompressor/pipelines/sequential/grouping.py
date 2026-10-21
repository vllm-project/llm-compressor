"""
Estimate the memory each node of a traced graph needs while its subgraph is calibrated,
so that `topological_partition` can group consecutive sequential targets into
subgraphs that fit in device memory
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.dist_utils import is_distributed
from loguru import logger
from torch.fx import Node
from torch.nn import Module

from llmcompressor.utils.dev import get_main_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from llmcompressor.args.dataset_arguments import DatasetArguments
    from llmcompressor.modifiers import Modifier

__all__ = ["plan_subgraph_memory", "log_partitions"]

GiB = 1024**3

# bytes of activations per token per hidden unit of one batch, from peaks measured on
# Qwen3-8B with SDPA at 2048 tokens: ~0.8 GiB at batch size 1 (estimated 0.78 GiB)
# and 2.8-4.9 GiB at batch size 8 (estimated 6.25 GiB)
ACTIVATION_BYTES = 100

# module name -> (onloaded weight and held modifier bytes, modifier scratch bytes)
ModuleCosts = dict[str, tuple[int, int]]

# largest batch size, longest sequence, and number of tokens including padding
InputShapes = tuple[int, int, int]


def plan_subgraph_memory(
    model: Module,
    node_modules: dict[Node, list[str]],
    dataloader: "DataLoader",
    modifiers: list["Modifier"],
    dataset_args: "DatasetArguments",
) -> tuple[dict[Node, int], int] | None:
    """
    Estimate the memory each node onloads and holds until its subgraph is compressed,
    and how much of it fits in a subgraph. Calibration activations cached on the
    device are taken out of the budget. Activations are freed before modules are
    compressed, so the larger of the activation estimate and the largest modifier
    scratch is reserved from the budget of every subgraph

    Called once persistent modules are onloaded, so the memory they use is already
    taken out of the budget

    :param model: model being calibrated
    :param node_modules: map of each node to the names of the modules it onloads
    :param dataloader: calibration data
    :param modifiers: modifiers which will calibrate the subgraphs
    :param dataset_args: arguments which set the budget and where activations are
        cached
    :return: map of node to bytes, and the bytes available to the nodes of a subgraph,
        or None to fall back to one target per subgraph
    """
    gpu_memory_utilization = dataset_args.gpu_memory_utilization
    device = get_main_device()
    if device.type == "cpu":
        logger.warning(
            'sequential_targets_per_subgraph="auto" needs an accelerator to measure '
            "memory. Falling back to one target per subgraph"
        )
        return None

    # weights are only decompressed when their subgraph runs, so neither their size
    # nor their shape can be read up front
    if dataset_args.layerwise_decompression:
        logger.warning(
            'sequential_targets_per_subgraph="auto" cannot estimate memory while '
            "weights are compressed (layerwise_decompression=True). Falling back to "
            "one target per subgraph"
        )
        return None

    # every rank must choose the same groups, since modifiers communicate when each
    # subgraph is compressed, so ranks agree on the inputs before estimating anything.
    # Whether activations are cached on a rank's device differs between ranks, so the
    # cache leaves each rank's budget before they agree on the smallest one
    budget = get_memory_budget(device, gpu_memory_utilization)
    shapes = get_input_shapes(dataloader)
    if budget is not None and shapes is not None:
        budget -= estimate_cached_activations(model, shapes, dataset_args, device)
    if is_distributed():
        budget, shapes = reduce_across_ranks(budget, shapes, device)

    if budget is None:
        logger.warning(
            f"Unable to measure free memory on {device}. Falling back to one target "
            "per subgraph"
        )
        return None

    activation_memory = estimate_activation_memory(model, shapes)
    if activation_memory is None:
        logger.warning(
            'sequential_targets_per_subgraph="auto" cannot estimate activation memory '
            "without `input_ids` in every batch and a `hidden_size` in the model "
            "config. Falling back to one target per subgraph"
        )
        return None

    names = {name for names in node_modules.values() for name in names}
    module_costs, unknown_modifiers = get_module_costs(
        model, names, modifiers, num_tokens=shapes[2]
    )
    if len(unknown_modifiers) > 0:
        logger.warning(
            'sequential_targets_per_subgraph="auto" cannot account for the memory '
            f"used by {', '.join(sorted(unknown_modifiers))}, which may grow with "
            "the number of targets per subgraph. Falling back to one target per "
            "subgraph"
        )
        return None

    largest_scratch = max((scratch for _, scratch in module_costs.values()), default=0)
    reserved = max(activation_memory, largest_scratch)
    if budget - reserved <= 0:
        logger.warning(
            f"No memory is left on {device} for subgraphs within "
            f"gpu_memory_utilization={gpu_memory_utilization} (budget "
            f"{budget / GiB:.2f} GiB, {reserved / GiB:.2f} GiB reserved for "
            "activations and compression). Falling back to one target per subgraph"
        )
        return None

    logger.info(
        f"Memory budget per subgraph {budget / GiB:.2f} GiB "
        f"(gpu_memory_utilization={gpu_memory_utilization}), of which "
        f"{reserved / GiB:.2f} GiB is reserved for activations and compression"
    )
    node_memory = {
        node: sum(module_costs[name][0] for name in set(names))
        for node, names in node_modules.items()
    }
    return node_memory, budget - reserved


def reduce_across_ranks(
    budget: int | None, shapes: InputShapes | None, device: torch.device
) -> tuple[int | None, InputShapes | None]:
    """
    Agree on the inputs to the estimate across ranks, which hold different calibration
    data and may see different free memory: the smallest budget and the largest input
    shapes. Values which any rank could not measure become None on every rank

    :param budget: this rank's memory budget in bytes, or None
    :param shapes: this rank's input shapes, or None
    :param device: device used to communicate between ranks
    :return: budget and input shapes shared by all ranks
    """
    # shapes are negated so that a single MIN reduction serves every value
    values = torch.tensor(
        [budget is not None, shapes is not None, budget or 0]
        + [-value for value in shapes or (0, 0, 0)],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MIN)

    has_budget, has_shapes, budget, *negated_shapes = values.tolist()
    return (
        budget if has_budget else None,
        tuple(-value for value in negated_shapes) if has_shapes else None,
    )


def log_partitions(
    partitions: list[list[Node]],
    target_nodes: set[Node],
    node_memory: dict[Node, int],
    max_memory: int,
):
    """
    Log the number of targets in each subgraph, and warn about nodes which do not fit
    in a subgraph even on their own

    :param partitions: partitions produced by `topological_partition`
    :param target_nodes: nodes which call sequential targets
    :param node_memory: map of node to bytes
    :param max_memory: bytes available to the nodes of a subgraph
    """
    sizes = [
        num_targets
        for partition in partitions
        if (num_targets := sum(node in target_nodes for node in partition)) > 0
    ]
    logger.info(
        f"Grouped {sum(sizes)} sequential targets into {len(sizes)} subgraphs of "
        f"sizes {sizes}"
    )

    oversized = [
        f"{node.name} ({memory / GiB:.2f} GiB)"
        for node, memory in node_memory.items()
        if memory > max_memory
    ]
    if len(oversized) > 0:
        logger.warning(
            f"{len(oversized)} nodes need more than the {max_memory / GiB:.2f} GiB "
            "available to a subgraph on their own, and may run out of memory: "
            f"{', '.join(oversized[:10])}"
        )


def get_module_costs(
    model: Module, names: set[str], modifiers: list["Modifier"], num_tokens: int
) -> tuple[ModuleCosts, set[str]]:
    """
    Get the memory each module onloads and holds until its subgraph is compressed, and
    the scratch memory needed to compress it, without onloading any weights

    :param model: model which owns the modules
    :param names: names of the modules to cost
    :param modifiers: modifiers which will calibrate the modules
    :param num_tokens: number of calibration tokens, including padding
    :return: map of module name to (weight and held, scratch) bytes, and the names of
        modifiers which could not estimate their memory
    """
    module_costs: ModuleCosts = {}
    unknown_modifiers: set[str] = set()
    with disable_onloading():
        for name in names:
            module = model.get_submodule(name)
            held, scratch = get_onload_bytes(module), 0
            for modifier in modifiers:
                estimate = modifier.estimate_sequential_memory(module, num_tokens)
                if estimate is None:
                    unknown_modifiers.add(type(modifier).__name__)
                    continue

                held += estimate[0]
                scratch = max(scratch, estimate[1])

            module_costs[name] = (held, scratch)

    return module_costs, unknown_modifiers


def get_onload_bytes(module: Module) -> int:
    """
    Get the device memory added by onloading a module's own parameters and buffers.
    Modules which are not offloaded, or are offloaded to their onload device, add
    nothing. Tied tensors count once per module, since each module is onloaded
    separately

    :param module: module whose tensors are counted (not including submodules)
    :return: number of bytes
    """
    parameters = module._parameters
    if not isinstance(parameters, OffloadCache):
        return 0
    if _is_same_device(parameters.offload_device, parameters.onload_device):
        return 0

    return sum(
        tensor.numel() * tensor.element_size()
        for cache in (module._parameters, module._buffers)
        for tensor in cache.offloaded_values.values()
        if tensor is not None
    )


def estimate_activation_memory(model: Module, shapes: InputShapes | None) -> int | None:
    """
    Estimate the activation memory of a subgraph as
    max_batch_size * max_seq_len * hidden_size * ACTIVATION_BYTES, since subgraphs
    run one batch at a time

    :param model: model being calibrated
    :param shapes: input shapes of the calibration data
    :return: bytes, or None if there are no input shapes or the model config has no
        `hidden_size`
    """
    hidden_size = get_hidden_size(model)
    if shapes is None or hidden_size is None:
        return None

    max_batch_size, max_seq_len, _ = shapes
    return max_batch_size * max_seq_len * hidden_size * ACTIVATION_BYTES


def estimate_cached_activations(
    model: Module,
    shapes: InputShapes,
    dataset_args: "DatasetArguments",
    device: torch.device,
) -> int:
    """
    Estimate the calibration activations which the pipeline caches on `device` between
    subgraphs: the hidden states of every token, kept twice when the sequential error
    is logged. Other cached values are not counted, such as position embeddings,
    attention masks, or intermediates wider than the hidden states when targets are
    smaller than decoder layers

    :param model: model being calibrated
    :param shapes: input shapes of the calibration data
    :param dataset_args: arguments which set where activations are cached
    :param device: device which subgraphs are onloaded to
    :return: bytes, or 0 if activations are cached on another device or the model
        config has no `hidden_size`
    """
    offload_device = torch.device(dataset_args.sequential_offload_device)
    hidden_size = get_hidden_size(model)
    if not _is_same_accelerator(offload_device, device) or hidden_size is None:
        return 0

    with disable_onloading():
        parameter = next(model.parameters(), None)
    value_bytes = 2 if parameter is None else parameter.element_size()
    copies = 2 if dataset_args.log_sequential_error else 1
    return copies * shapes[2] * hidden_size * value_bytes


def get_hidden_size(model: Module) -> int | None:
    """
    :param model: model being calibrated
    :return: the hidden size of the (text) model, or None if the config has none
    """
    config = getattr(model, "config", None)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    hidden_size = getattr(config, "hidden_size", None)
    return hidden_size if isinstance(hidden_size, int) else None


def get_input_shapes(dataloader: "DataLoader") -> InputShapes | None:
    """
    Read the shapes of the calibration batches from the dataloader, without running
    the model

    :param dataloader: calibration data
    :return: largest batch size, longest sequence and number of tokens including
        padding of the batches' `input_ids`, or None if a batch has no 2D `input_ids`
    """
    max_batch_size, max_seq_len, num_tokens = 0, 0, 0

    # shuffling samplers draw from the global random state on every pass, so it is
    # restored afterwards: the next pass then calibrates on these same batches
    with torch.random.fork_rng(devices=[]):
        for batch in dataloader:
            input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
            if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
                return None

            batch_size, seq_len = input_ids.shape
            max_batch_size = max(max_batch_size, batch_size)
            max_seq_len = max(max_seq_len, seq_len)
            num_tokens += batch_size * seq_len

    return max_batch_size, max_seq_len, num_tokens


def get_memory_budget(
    device: torch.device, gpu_memory_utilization: float
) -> int | None:
    """
    Get the device memory available to a subgraph: the fraction of total memory that
    calibration may use, less what is already in use by this or other processes

    :param device: device which subgraphs are onloaded to
    :param gpu_memory_utilization: fraction of the device's total memory that
        calibration may use
    :return: bytes, or None if memory cannot be measured
    """
    try:
        torch.accelerator.empty_cache()
        free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
    except (RuntimeError, NotImplementedError):
        return None

    return int(free_bytes - (1 - gpu_memory_utilization) * total_bytes)


def _is_same_device(device: Any, other: Any) -> bool:
    try:
        return torch.device(device) == torch.device(other)
    except (RuntimeError, TypeError):
        return False


def _is_same_accelerator(device: torch.device, other: torch.device) -> bool:
    """Whether two devices are the same, where a device without an index (e.g.
    "cuda") is the current device of its type"""
    return device.type == other.type and (
        device.index is None or other.index is None or device.index == other.index
    )
