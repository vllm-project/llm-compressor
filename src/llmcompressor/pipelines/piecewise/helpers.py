import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from torch.fx import Graph, GraphModule, Node
from torch.nn import Module
from transformers.utils.fx import HFTracer

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import calibration_forward_context, disable_hf_hook
from llmcompressor.utils.pytorch.module import get_no_split_params


@dataclass
class Subgraph:
    graph: Graph
    input_names: List[str]
    consumed_names: List[str]


__all__ = ["get_compression_targets", "trace_subgraphs"]


def get_compression_targets(model: Module, recipe: Recipe) -> Set[Module]:
    """
    TODO: true sequential

    List of modules which are guaranteed to be split into different partitions and
    whose inner operations will not be traced
    """
    no_split_params = get_no_split_params(model)
    return set(
        module for module in model.modules() if type(module).__name__ in no_split_params
    )


def trace_subgraphs(
    model: Module, sample_input: Dict[str, Any], targets: Set[Module]
) -> List[Subgraph]:
    # initialize arguments
    tracer = get_tracer(targets)
    concrete_args = populate_concrete_args(model, sample_input)

    # trace
    with (
        calibration_forward_context(model),
        HooksMixin.disable_hooks(),
        disable_hf_hook(model, recurse=True),
    ):
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

    return subgraphs


def get_tracer(targets: List[Module]) -> HFTracer:
    class PiecewiseTracer(HFTracer):
        def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
            if module in targets:
                return True  # Treat as leaf, skip tracing inside this module
            return super().is_leaf_module(module, module_qualified_name)

    return PiecewiseTracer()


def populate_concrete_args(model: Module, sample_input: Dict) -> Dict:
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


def get_target_nodes(graph: GraphModule, targets: Set[Module]) -> Set[Node]:
    return set(
        node
        for node in graph.graph.nodes
        if node.op == "call_module" and graph.get_submodule(node.target) in targets
    )


def check_assumption(graph: Graph) -> bool:
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


def topological_partition(graph: GraphModule, targets: Set[Module]) -> List[List[Node]]:
    assert check_assumption(graph.graph)
    target_nodes = get_target_nodes(graph, targets)

    partitions: List[List[Node]] = [[]]
    remaining_indegrees = {
        node: len([node for node in node.all_input_nodes if node.op != "get_attr"])
        for node in graph.graph.nodes
    }
    partition_index = 0  # global counter

    # start with graph input nodes
    queue = deque(
        node
        for node in graph.graph.nodes
        if remaining_indegrees[node] == 0 and node.op != "get_attr"
    )
    while len(queue) > 0:
        node = queue.popleft()

        # guarantee targets are assigned to disjoint partitions
        if node in target_nodes:
            partition_index += 1
            partitions.append([])

        # assign to partition
        partitions[partition_index].append(node)

        # recurse on last indegree only in order to guarantee that
        # the node is assigned to maximal partition
        for user in node.users:
            remaining_indegrees[user] -= 1
            if remaining_indegrees[user] == 0:
                queue.append(user)

    # a perfect solution would involve implicitly consolodating partition indices so
    # that each node is assigned to the maximum partition possible (in order to delay
    # execution as long as possible), but this covers the most costly case (get_attr)
    for node in graph.graph.nodes:
        if node.op == "get_attr":
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
            # if node.op != "get_attr"
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

        # Save the subgraph for this partition
        graph.lint()
        input_names = [node.name for node in graph.nodes if node.op == "placeholder"]
        subgraphs.append(
            Subgraph(
                graph=graph,
                input_names=input_names,
                consumed_names=[],  # populated later
            )
        )

        assert check_assumption(graph)

    return subgraphs


def trace_consumed_names(subgraphs: List[Dict[str, Any]]):
    # TODO: update consumed names as new partitions are appended
    # populate consumed_names according to when inputs are last used
    # in order to vacate the `intermediates` cache and save memory
    all_input_names = set().union(*(subgraph.input_names for subgraph in subgraphs))
    for input_name in all_input_names:
        for subgraph in reversed(subgraphs):
            if input_name in subgraph.input_names:
                subgraph.consumed_names.append(input_name)
                break
        else:
            assert False
