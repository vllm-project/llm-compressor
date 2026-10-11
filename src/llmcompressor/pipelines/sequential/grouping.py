"""
Group consecutive sequential targets into subgraphs that fit in device memory, from
the memory each node of the traced graph adds while its subgraph is calibrated
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.dist_utils import is_distributed
from loguru import logger
from torch.fx import GraphModule, Node
from torch.nn import Module

from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    Subgraph,
)

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = [
    "SubgraphMemory",
    "plan_subgraph_memory",
    "log_subgraph_memory",
    "count_calibration_tokens",
]

GiB = 1024**3

# bytes of activations per token per hidden unit of one batch, from peaks measured on
# Qwen3-8B with SDPA at 2048 tokens: ~0.8 GiB at batch size 1 (estimated 0.78 GiB)
# and 2.8-4.9 GiB at batch size 8 (estimated 6.25 GiB)
ACTIVATION_BYTES = 100

# module name -> (onloaded weight bytes, held modifier bytes, modifier scratch bytes)
ModuleCosts = dict[str, tuple[int, int, int]]


class SubgraphMemory:
    """
    Memory check passed to `topological_partition`. A partition costs the onloaded
    weights and held modifier memory of every module its nodes call, plus the larger
    of the activation estimate and the largest modifier scratch, since activations are
    freed before modules are compressed. Each module counts once per partition

    :param model: model being calibrated
    :param module_costs: map of module name to (weight, held, scratch) bytes
    :param activation_memory: estimated activation memory of a subgraph, in bytes
    :param budget: device memory available to a subgraph, in bytes
    """

    def __init__(
        self,
        model: Module,
        module_costs: ModuleCosts,
        activation_memory: int,
        budget: int,
    ):
        self.model = model
        self.module_costs = module_costs
        self.activation_memory = activation_memory
        self.budget = budget
        self._node_modules: dict[Node, list[str]] = {}

        # running cost of the partition being filled
        self._partition: list[Node] | None = None
        self._names: set[str] = set()
        self._held = 0
        self._scratch = 0
        self._num_seen = 0

    def __call__(self, partition: list[Node], chunk: list[Node]) -> bool:
        """
        :param partition: nodes already in the partition being filled
        :param chunk: nodes from the next target up to the target after it
        :return: True if the chunk does not fit in the partition
        """
        if partition is not self._partition:
            self._partition, self._names = partition, set()
            self._held, self._scratch, self._num_seen = 0, 0, 0

        # account for the nodes added to the partition since the last call
        new_names = self._new_names(partition[self._num_seen :])
        added_held, added_scratch = self.get_cost(new_names)
        self._names |= new_names
        self._held += added_held
        self._scratch = max(self._scratch, added_scratch)
        self._num_seen = len(partition)

        added_held, added_scratch = self.get_cost(self._new_names(chunk))
        held = self._held + added_held
        scratch = max(self._scratch, added_scratch)
        return held + max(self.activation_memory, scratch) > self.budget

    def _new_names(self, nodes: list[Node]) -> set[str]:
        return {
            name for node in nodes for name in self.node_modules(node)
        } - self._names

    def get_cost(self, names: set[str]) -> tuple[int, int]:
        """
        :param names: names of the modules in a partition
        :return: (weights + held modifier memory, largest scratch) in bytes
        """
        held = sum(
            self.module_costs[name][0] + self.module_costs[name][1] for name in names
        )
        scratch = max((self.module_costs[name][2] for name in names), default=0)
        return held, scratch

    def get_peak(self, names: set[str]) -> int:
        """
        :param names: names of the modules in a partition
        :return: estimated peak memory of the partition in bytes
        """
        held, scratch = self.get_cost(names)
        return held + max(self.activation_memory, scratch)

    def node_modules(self, node: Node) -> list[str]:
        """
        :param node: node of the traced graph
        :return: names of the modules the node onloads when its subgraph runs
        """
        if node not in self._node_modules:
            self._node_modules[node] = get_node_modules(self.model, node)
        return self._node_modules[node]


def plan_subgraph_memory(
    model: Module,
    graph: GraphModule,
    activations: IntermediatesCache,
    modifiers: list["Modifier"],
    device: torch.device,
    gpu_memory_utilization: float,
) -> SubgraphMemory | None:
    """
    Estimate what each part of the graph costs and how much memory a subgraph may use.
    Called after persistent modules are onloaded and the intermediates cache is built,
    so the memory they use is already taken out of the budget

    :param model: model being calibrated
    :param graph: traced graph which will be partitioned
    :param activations: cache holding the calibration batches
    :param modifiers: modifiers which will calibrate the subgraphs
    :param device: device which subgraphs are onloaded to
    :param gpu_memory_utilization: fraction of the device's total memory that
        calibration may use
    :return: memory check for `topological_partition`, or None to fall back to one
        target per subgraph when memory cannot be estimated
    """
    if device.type == "cpu":
        logger.warning(
            'sequential_targets_per_subgraph="auto" needs an accelerator to measure '
            "memory. Falling back to one target per subgraph"
        )
        return None

    names = {
        name for node in graph.graph.nodes for name in get_node_modules(model, node)
    }
    module_costs, unknown_modifiers = get_module_costs(model, names, modifiers)
    if len(unknown_modifiers) > 0:
        logger.warning(
            'sequential_targets_per_subgraph="auto" cannot account for the memory '
            f"used by {', '.join(sorted(unknown_modifiers))}, which may grow with "
            "the number of targets per subgraph. Falling back to one target per "
            "subgraph"
        )
        return None

    activation_memory = estimate_activation_memory(model, activations)
    if activation_memory is None:
        logger.warning(
            'sequential_targets_per_subgraph="auto" cannot estimate activation memory '
            "without `input_ids` in every batch and a `hidden_size` in the model "
            "config. Falling back to one target per subgraph"
        )
        return None

    budget = get_memory_budget(device, gpu_memory_utilization)
    if budget is None:
        logger.warning(
            f"Unable to measure free memory on {device}. Falling back to one target "
            "per subgraph"
        )
        return None

    # every rank must choose the same groups, since modifiers communicate when
    # each subgraph is compressed
    if is_distributed():
        budget, activation_memory, module_costs = reduce_across_ranks(
            budget, activation_memory, module_costs, device
        )

    # checked after the all-reduce so that every rank falls back together
    if budget <= 0:
        logger.warning(
            f"No memory is left on {device} for subgraphs within "
            f"gpu_memory_utilization={gpu_memory_utilization} (budget "
            f"{budget / GiB:.2f} GiB). Falling back to one target per subgraph"
        )
        return None

    logger.info(
        f"Memory budget per subgraph {budget / GiB:.2f} GiB "
        f"(gpu_memory_utilization={gpu_memory_utilization}), estimated activation "
        f"memory {activation_memory / GiB:.2f} GiB"
    )
    return SubgraphMemory(model, module_costs, activation_memory, budget)


def reduce_across_ranks(
    budget: int,
    activation_memory: int,
    module_costs: ModuleCosts,
    device: torch.device,
) -> tuple[int, int, ModuleCosts]:
    """
    Agree on the inputs to grouping across ranks, which hold different calibration
    data and may see different free memory: the smallest budget, and the largest
    activation estimate and module costs

    :param budget: this rank's memory budget in bytes
    :param activation_memory: this rank's activation estimate in bytes
    :param module_costs: this rank's map of module name to (weight, held, scratch)
    :param device: device used to communicate between ranks
    :return: budget, activation estimate and module costs shared by all ranks
    """
    # estimates are negated so that a single MIN reduction serves every value
    names = sorted(module_costs)
    values = torch.tensor(
        [budget, -activation_memory]
        + [-cost for name in names for cost in module_costs[name]],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MIN)

    costs = (-values[2:]).view(-1, 3).tolist()
    return (
        int(values[0]),
        -int(values[1]),
        {name: tuple(cost) for name, cost in zip(names, costs)},
    )


def log_subgraph_memory(
    memory: SubgraphMemory,
    subgraphs: list[Subgraph],
    model: Module,
    targets: set[Module],
):
    """
    Log the number of targets in each subgraph, and warn about subgraphs whose
    estimated peak exceeds the budget

    :param memory: memory check the subgraphs were partitioned with
    :param subgraphs: subgraphs in order of execution
    :param model: model which owns the subgraphs
    :param targets: sequential target modules
    """
    sizes, over_budget = [], []
    for index, subgraph in enumerate(subgraphs):
        modules = subgraph.submodule_dict(model)
        num_targets = sum(
            1
            for node in subgraph.graph.find_nodes(op="call_module")
            if model.get_submodule(node.target) in targets
        )
        if num_targets > 0:
            sizes.append(num_targets)
        peak = memory.get_peak(set(modules))
        if peak > memory.budget:
            over_budget.append(f"{index} ({peak / GiB:.2f} GiB)")

    logger.info(
        f"Grouped {sum(sizes)} sequential targets into {len(sizes)} subgraphs of "
        f"sizes {sizes}"
    )
    if len(over_budget) > 0:
        logger.warning(
            f"{len(over_budget)} subgraphs need more than the memory budget of "
            f"{memory.budget / GiB:.2f} GiB even with a single target, and may run "
            f"out of memory: {', '.join(over_budget[:10])}"
        )


def get_node_modules(model: Module, node: Node) -> list[str]:
    """
    Get the modules a node onloads when its subgraph runs: the module it calls and
    that module's submodules, plus modules recorded as run inside wrapped code

    :param model: model which owns the graph
    :param node: node of the traced graph
    :return: module names
    """
    roots = list(node.meta.get(INVOKED_SUBMODULES_META_KEY, ()))
    if node.op == "call_module":
        roots.insert(0, node.target)

    names = []
    for root in roots:
        for relative_name, _ in model.get_submodule(root).named_modules():
            names.append(f"{root}.{relative_name}" if relative_name else root)

    return names


def get_module_costs(
    model: Module, names: set[str], modifiers: list["Modifier"]
) -> tuple[ModuleCosts, set[str]]:
    """
    Get the onloaded weight bytes and modifier memory of each module, without
    onloading any weights

    :param model: model which owns the modules
    :param names: names of the modules to cost
    :param modifiers: modifiers which will calibrate the modules
    :return: map of module name to (weight, held, scratch) bytes, and the names of
        modifiers which could not estimate their memory
    """
    module_costs: ModuleCosts = {}
    unknown_modifiers: set[str] = set()
    with disable_onloading():
        for name in names:
            module = model.get_submodule(name)
            held, scratch = 0, 0
            for modifier in modifiers:
                estimate = modifier.estimate_sequential_memory(module)
                if estimate is None:
                    unknown_modifiers.add(type(modifier).__name__)
                    continue

                held += estimate[0]
                scratch = max(scratch, estimate[1])

            module_costs[name] = (get_onload_bytes(module), held, scratch)

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


def estimate_activation_memory(
    model: Module, activations: IntermediatesCache
) -> int | None:
    """
    Estimate the activation memory of a subgraph as
    max_batch_size * max_seq_len * hidden_size * ACTIVATION_BYTES, since subgraphs
    run one batch at a time

    :param model: model being calibrated
    :param activations: cache holding the calibration batches
    :return: bytes, or None if the cache has no `input_ids` or the model config has
        no `hidden_size`
    """
    config = getattr(model, "config", None)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        return None

    shapes = get_input_shapes(activations)
    if shapes is None:
        return None

    max_batch_size = max((batch_size for batch_size, _ in shapes), default=0)
    max_seq_len = max((seq_len for _, seq_len in shapes), default=0)
    return max_batch_size * max_seq_len * hidden_size * ACTIVATION_BYTES


def count_calibration_tokens(activations: IntermediatesCache) -> int | None:
    """
    Count the calibration tokens held by the cache, including padding

    :param activations: cache holding the calibration batches
    :return: number of tokens, or None if the cache has no `input_ids`
    """
    shapes = get_input_shapes(activations)
    if shapes is None:
        return None

    return sum(batch_size * seq_len for batch_size, seq_len in shapes)


def get_input_shapes(activations: IntermediatesCache) -> list[tuple[int, int]] | None:
    """
    :param activations: cache holding the calibration batches
    :return: (batch size, sequence length) of each batch's `input_ids`, or None if a
        batch has no 2D `input_ids`
    """
    shapes = []
    for batch in activations.batch_intermediates:
        value = getattr(batch.get("input_ids"), "value", None)
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            return None
        shapes.append((value.shape[0], value.shape[1]))

    return shapes


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
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(
            "gpu_memory_utilization must be greater than 0 and at most 1, got "
            f"{gpu_memory_utilization}"
        )

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
