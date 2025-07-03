import contextlib
import inspect
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from compressed_tensors.quantization import find_name_or_class_matches
from compressed_tensors.utils import (
    has_offloaded_params,
    offloaded_dispatch,
    remove_dispatch,
)
from loguru import logger
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import PythonCode
from torch.fx.proxy import Argument
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.fx import HFTracer

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.helpers import calibration_forward_context, patch_attr
from llmcompressor.utils.pytorch.module import get_no_split_params

from .ast_helpers import autowrap_forwards

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = [
    "trace_subgraphs",
    "Subgraph",
    "get_sequential_targets",
    "dispatch_for_sequential",
]


@dataclass
class Subgraph:
    """
    Dataclass specifying an executable subgraph of a model graph

    :param graph: subgraph of model graph
    :param input_names: argument names of the compiled forward function
    :param consumed_names: argument names which are not used by any subsequent subgraphs
        and can therefore be deleted from the intermediates cache
    """

    graph: Graph
    input_names: Set[str]
    consumed_names: Set[str]
    _code: Optional[PythonCode] = None

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the operations within the subgraph

        :param \\*args: argument inputs to subgraph forward function
        :param \\**kwargs: keyword inputs to subgraph forward function
        :return keyword outputs of subgraph forward function (non-consumed variables):
        """
        if self._code is None:
            self._code = self.graph.python_code("self")
            exec(self._code.src, self._code.globals)

        forward_fn = self._code.globals.get("forward")

        try:
            outputs = forward_fn(*args, **kwargs)
        except Exception as exception:
            raise RuntimeError(
                "Raised an exception during execution of the following code:\n"
                f"```\n{add_line_numbers(self._code.src)}\n```\n"
                "This is likely due to a violation of shape assumptions made when "
                "tracing"
            ) from exception

        return outputs


def trace_subgraphs(
    model: PreTrainedModel,
    sample_input: Dict[str, Any],
    sequential_targets: List[str],
    ignore: List[str],
) -> List[Subgraph]:
    """
    Trace a model to produce subgraphs, where each sequential target belongs to exactly
    one subgraph and where executing each subgraph in order is equivalent to executing
    the original model

    :param model: model being traced
    :param sample_input: inputs whose values will change during execution but whose
        __len__, __bool__, and __contains__ values are assumed constant across batches
    :param sequential_targets: list of patterns matching sequential targets
    :param ignore: function and method names to skip during tracing
    :return: a list of Subgraphs in order of execution
    """
    # find modules
    targets = match_modules(model, sequential_targets)
    ancestors = get_sequential_ancestors(model, targets)
    offloaded = set(m for m in model.modules() if has_offloaded_params(m))

    # initialize arguments
    tracer = SequentialTracer(ancestors, offloaded)
    concrete_args = populate_concrete_args(model, sample_input)

    with contextlib.ExitStack() as stack:
        # calibration context
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(HooksMixin.disable_hooks())

        # flags useful for tracing
        stack.enter_context(patch_attr(model.config, "_attn_implementation", "eager"))
        stack.enter_context(patch_attr(torch.compiler, "_is_compiling_flag", True))

        # autowrap forwards
        stack.enter_context(autowrap_forwards(ancestors, ignore))
        stack.enter_context(patch_attr(type(model), "forward", model.forward.__func__))

        graph = GraphModule(
            model,
            tracer.trace(
                model,
                dummy_inputs=sample_input,
                concrete_args=concrete_args,
                complete_concrete_args_with_inputs_not_in_dummy_inputs=False,
                # bug in trace throws an error for variadic
                # args and kwargs in function signature
            ),
        )

    # copy metadata
    graph.config = model.config
    graph.class_for_deserialization = model.__class__
    graph.device = model.device

    # perform subgraph partition
    partitions = topological_partition(graph, targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    # As currently implemented, `topological_partition` generates an extra subgraph at
    # the beginning which does not contain a target. This adds a little more runtime,
    # and could be folded into the first subgraph in the future
    if len(subgraphs) != len(targets) + 1:
        logger.warning(
            f"Expected {len(targets)} subgraphs, but only traced {len(subgraphs)}. "
            "This is likely due to having wrapped code which calls sequential targets"
        )

    return subgraphs


class SequentialTracer(HFTracer):
    """
    Get a tracer specialized for the given model. The resulting tracer will not trace
    inside of sequential targets, nor any modules which are not call graph ancestors of
    sequential targets

    Tracing within sequential targets is unnecessary, and tracing within offloaded
    modules may result in meta tensors being added to the model graph

    :param ancestors: modules which are ancestors of sequential targets
    :param offloaded: modules which have offloaded params and should not be traced
    """

    def __init__(self, ancestors: Set[Module], offloaded: Set[Module]):
        self.ancestors = ancestors
        self.offloaded = offloaded

        # skip any mask creation functions not already caught by the autowrapper
        super().__init__(autowrap_functions=_get_autowrap_functions())

        # check unlikely case that ancestors have direct params which are offloaded
        offloaded_ancestors = offloaded & ancestors
        if offloaded_ancestors:
            names = set(module.__class__.__name__ for module in offloaded_ancestors)
            logger.warning(
                "The following modules are call graph ancestors of sequential targets,"
                f"but also contain offloaded modules: {names}.\n"
                "These modules will not be traced, and any sequential target children "
                "will be executed jointly, which may lead to OOM errors"
            )

    def create_arg(self, a: Any) -> Argument:
        # special extension allows models which depend on config values to be traced
        if isinstance(a, PretrainedConfig):
            kwargs = {k: self.create_arg(v) for k, v in a.to_dict().items()}
            return self.create_node("call_function", a.__class__, (), kwargs)

        else:
            return super().create_arg(a)

    def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
        # do not trace non-ancestors or modules with offloaded params
        return module not in self.ancestors or module in self.offloaded


def populate_concrete_args(model: Module, sample_input: Dict) -> Dict:
    """
    Creates concrete args which, unlike the equivalent function provided by
    transformers.utils.fx, creates default values for variadic arguments, which are
    needed by some models.

    :param model: model being traced
    :param sample_input: values used to symbolically trace the model. All arguments
        to the model.forward function which are not in the sample_input are considered
        concrete args
    :return: dictionary mapping concrete argument names to their default values
    """
    sig = inspect.signature(model.forward)

    concrete_args = {}
    for parameter in sig.parameters.values():
        if parameter.name in sample_input:
            continue
        if parameter.kind == inspect._ParameterKind.VAR_POSITIONAL:
            value = list()
        elif parameter.kind == inspect._ParameterKind.VAR_KEYWORD:
            value = dict()
        elif parameter.name == "use_cache":
            value = False
        else:
            value = parameter.default

        concrete_args[parameter.name] = value

    return concrete_args


def find_target_nodes(graph: GraphModule, targets: Set[Module]) -> Set[Node]:
    """
    Find all nodes whose execution is equivalent to executing the target modules.
    Note that these nodes are guaranteed to be treated as leaf nodes by SequentialTracer

    :param graph: graph containing target nodes
    :param targets: modules whose nodes are being searched for
    :return: set of all nodes which call the target modules
    """
    return set(
        node
        for node in graph.graph.nodes
        if node.op == "call_module" and graph.get_submodule(node.target) in targets
    )


def topological_partition(graph: GraphModule, targets: Set[Module]) -> List[List[Node]]:
    """
    Partition the graph into partitions such that each `target` belongs to exactly one
    partition and executing each partition depends only on intermediate values produced
    by executing the partitions before it.

    :param graph: graph being partitioned
    :param targets: target modules which will be assigned to disjoint partitions
    :return: list of partitions, where each partition is a list of nodes belonging to
        that partition
    """
    assert graph_is_well_formed(graph.graph)
    target_nodes = find_target_nodes(graph, targets)

    partitions: List[List[Node]] = [[]]
    remaining_indegrees = {
        node: len([node for node in node.all_input_nodes if node.op != "get_attr"])
        for node in graph.graph.nodes
    }
    partition_index = 0  # global counter

    # start with graph input nodes,
    # but delay the `get_attr` nodes as long as possible
    queue = deque(
        node
        for node in graph.graph.nodes
        if remaining_indegrees[node] == 0 and node.op != "get_attr"
    )
    while len(queue) > 0:
        node = queue.popleft()

        # assign to partition
        partitions[partition_index].append(node)

        # guarantee targets are assigned to disjoint partitions
        if node in target_nodes:
            partition_index += 1
            partitions.append([])

        # recurse on last indegree only in order to guarantee that
        # the node is assigned to maximal partition
        for user in node.users:
            remaining_indegrees[user] -= 1
            if remaining_indegrees[user] == 0:
                queue.append(user)

    # an ideal implementation would involve implicitly consolidating partition indices
    # so that each node is assigned to the maximum partition possible (in order to delay
    # execution as long as possible), but saving these nodes for last covers the most
    # common and costly case (get_attr)
    for node in graph.graph.find_nodes(op="get_attr"):
        user_partitions = []
        for user in node.users:
            for index in range(len(partitions)):
                if user in partitions[index]:
                    user_partitions.append(index)
                    break
        partition_index = min(user_partitions)
        partitions[partition_index].insert(0, node)

    assert set().union(*partitions) == set(graph.graph.nodes)
    return partitions


def partition_graph(model: Module, partitions: List[List[Node]]) -> List[Subgraph]:
    """
    Convert each partition into a Subgraph. Each Subgraph returns a dictionary mapping
    of output node names to their computed values. Note that the `consumed_names`
    attribute of each Subgraph remains empty, to be later populated by
    `trace_consumed_names`

    :param model: model which owns the produced Subgraphs
    :param partitions: list of partitions, where each partition is a list of nodes
        belonging to that partition
    :return: list of subgraphs in order of execution
    """
    subgraphs = []

    # create subgraphs
    for partition_nodes in partitions:
        # create a new graph for the partition
        graph = Graph(model)
        node_map = {}

        # add placeholders for inputs not in this subgraph. use set to deduplicate
        new_input_nodes = {
            input_node
            for node in partition_nodes
            for input_node in node.all_input_nodes
            if input_node not in partition_nodes and input_node.op
        }
        for input_node in new_input_nodes:
            node_map[input_node] = graph.placeholder(input_node.name)

        # add the nodes to subgraph
        for node in partition_nodes:
            node_map[node] = graph.node_copy(node, lambda n: node_map[n])

        # add an output node to collect all subgraph outputs into a dictionary
        if len(graph.find_nodes(op="output")) <= 0:
            output_dict = {
                node.name: node_map[node]
                for node in partition_nodes
                if any(user not in partition_nodes for user in node.users.keys())
            }
            graph.output(output_dict)

        # save the subgraph for this partition
        graph.lint()
        input_names = set(node.name for node in graph.nodes if node.op == "placeholder")
        subgraphs.append(
            Subgraph(
                graph=graph,
                input_names=input_names,
                consumed_names=set(),  # populated later
            )
        )

        assert graph_is_well_formed(graph)

    return subgraphs


def trace_consumed_names(subgraphs: List[Subgraph]):
    """
    Populate the `consumed_names` attribute of each Subgraph according to when inputs
    are last used in order to vacate the `intermediates` cache and save memory

    :param subgraphs: list of subgraphs with empty `consumed_names` attributes
    """
    # populate consumed_names according to when inputs are last used
    # in order to vacate the `intermediates` cache and save memory
    all_input_names = set().union(*(subgraph.input_names for subgraph in subgraphs))
    for input_name in all_input_names:
        for subgraph in reversed(subgraphs):
            if input_name in subgraph.input_names:
                subgraph.consumed_names.add(input_name)
                break
        else:
            raise ValueError(f"Could not find input name {input_name} in subgraphs")


def graph_is_well_formed(graph: Graph) -> bool:
    """
    A graph is well formed if and only if
    `nodeA in NodeB.users <=> nodeB in Node.A.all_input_nodes`

    :param graph: graph being checked
    :return: True if the graph is well formed, False otherwise
    """
    for node in graph.nodes:
        for user in node.users:
            if node not in user.all_input_nodes:
                return False

        for input_node in node.all_input_nodes:
            if node not in input_node.users:
                return False

        if len(node.users) != len(set(node.users)) or len(node.all_input_nodes) != len(
            set(node.all_input_nodes)
        ):
            return False

    return True


def match_modules(model: Module, target_names: List[str]) -> Set[Module]:
    """
    Find modules whose names match the patterns given by `target_names`

    :param model: model containing submodules to find
    :param target_names: target patterns to find
    :return: all submodules matching `target_names`
    """
    return set(
        module
        for name, module in model.named_modules()
        if find_name_or_class_matches(name, module, target_names)
    )


def get_sequential_targets(
    modifiers: List[Modifier], model: PreTrainedModel, args: "DatasetArguments"
) -> List[str]:
    """
    Infer sequential targets from modifiers list and dataset args

    :param model: model being calibrated
    :param modifiers: list of modifiers being applied during calibration
    :param dataset_args: dataset arguments passed by user
    :return: list of sequential targets
    """
    modifier_targets = [
        (modifier, modifier.sequential_targets)
        for modifier in modifiers
        if getattr(modifier, "sequential_targets", None) is not None
    ]

    # deprecation warning
    if len(modifier_targets) >= 1:
        logger.warning(
            "Passing sequential targets through modifiers is deprecated, "
            "please use `oneshot(sequential_targets=...)`"
        )

    # cannot infer from multiple modifiers
    if len(modifier_targets) >= 2:
        types = [type(modifier) for modifier, _ in modifier_targets]
        raise ValueError(
            "Cannot infer sequential targets from multiple sequential modifiers "
            f"({types})"
        )

    # resolve single modifier
    if len(modifier_targets) == 1:
        if args.sequential_targets is not None:
            raise ValueError(
                f"Got sequential targets from both {type(modifier_targets[0][0])} "
                "and dataset arguments `sequential_targets`"
            )

        sequential_targets = modifier_targets[0][1]

    # if no modifiers, use data args
    else:
        sequential_targets = args.sequential_targets  # may be `None`

    # validate and infer
    if sequential_targets is None:
        return get_no_split_params(model)
    elif isinstance(sequential_targets, str):
        return [sequential_targets]
    else:
        return sequential_targets


def add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    numbered_lines = [f"{i + 1} {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def get_sequential_ancestors(model: Module, targets: Set[Module]) -> Set[Module]:
    """
    Find modules which are call graph ancestors of the given sequential targets

    :param model: model containing sequential targets
    :param targets: sequential targets to find ancestors of
    :return: call graph ancestors of sequential targets
    """
    ancestors = set()

    def is_ancestor(module: Module) -> bool:
        if module in ancestors or module in targets:
            return True

        # eagerly compute list in order to avoid early stopping and :. missing ancestors
        _is_ancestor = any([is_ancestor(child) for child in module.children()])
        if _is_ancestor:
            ancestors.add(module)

        return _is_ancestor

    is_ancestor(model)
    return ancestors


def dispatch_for_sequential(model: PreTrainedModel) -> PreTrainedModel:
    """
    Dispatch a model for sequential calibration using a sequential pipeline.
    The model will be offloaded to the CPU and dispatched to CUDA device if available.
    Removes any existing hooks.

    :param model: model to dispatch
    :return: dispatched model
    """
    remove_dispatch(model)

    if torch.cuda.is_available():
        offloaded_dispatch(model, execution_device=torch.device("cuda:0"))
    else:
        logger.warning("CUDA is not available! Compressing model on CPU instead")

    return model


def _get_autowrap_functions() -> Tuple[Callable[[Any], Any], ...]:
    try:
        from transformers.masking_utils import LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING

        return tuple(LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.values())
    except ImportError:
        return tuple()
