import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from compressed_tensors import has_offloaded_params
from compressed_tensors.quantization import find_name_or_class_matches
from loguru import logger
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import PythonCode
from torch.fx.proxy import Argument
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.fx import HFTracer

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.helpers import calibration_forward_context, patch_attr

__all__ = ["trace_subgraphs", "Subgraph"]


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
    :param ignore: TODO: unused, in the future will specify functions and methods to
        skip during tracing
    :return: a list of Subgraphs in order of execution
    """
    # find modules
    sequential_targets = match_modules(model, sequential_targets)

    # initialize arguments
    tracer = get_tracer(model, sequential_targets)
    concrete_args = populate_concrete_args(model, sample_input)

    # trace
    with calibration_forward_context(model), HooksMixin.disable_hooks():
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
    partitions = topological_partition(graph, sequential_targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    return subgraphs


def get_tracer(model: Module, sequential_targets: Set[Module]) -> HFTracer:
    """
    Get a tracer specialized for the given model. The resulting tracer will not trace
    inside of sequential targets, nor any modules which are not call graph ancestors of
    sequential targets

    Tracing within sequential targets is unnecessary, and tracing within offloaded
    modules may result in meta tensors being added to the model graph

    :param model: model being traced
    :param sequential_targets: modules which are sequential targets
    """
    sequential_ancestors = get_sequential_ancestors(model, sequential_targets)
    offloaded_modules = set(m for m in model.modules() if has_offloaded_params(m))

    # check unlikely case that ancestors have direct params which are offloaded
    offloaded_ancestors = offloaded_modules & sequential_ancestors
    if offloaded_ancestors:
        names = set(module.__class__.__name__ for module in offloaded_ancestors)
        logger.warning(
            "The following modules are call graph ancestors of sequential targets,"
            f"but also contain offloaded modules: {names}.\n"
            "These modules will not be traced, and any sequential target children will "
            "be executed jointly, which may lead to OOM errors"
        )

    class SequentialTracer(HFTracer):
        def create_arg(self, a: Any) -> Argument:
            # special extension allows models which depend on config values to be traced
            if isinstance(a, PretrainedConfig):
                kwargs = {k: self.create_arg(v) for k, v in a.to_dict().items()}
                return self.create_node("call_function", a.__class__, (), kwargs)

            else:
                return super().create_arg(a)

        def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
            return module not in sequential_ancestors or module in offloaded_modules

        def trace(self, root: Union[Module, Callable], *args, **kwargs) -> Graph:
            if isinstance(root, Module):
                # due to a bug in Tracer.create_args_for_root (_patch_function),
                # we must unwrap function wrappers prior to tracing, for example
                # the `deprecate_kwarg` by transformers which wraps forward
                unwrapped_forward = inspect.unwrap(type(root).forward)

                # we override the class method because the
                # class method is the one being traced
                with patch_attr(type(root), "forward", unwrapped_forward):
                    return super().trace(root, *args, **kwargs)

            else:
                return super().trace(root, *args, **kwargs)

    return SequentialTracer()


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
