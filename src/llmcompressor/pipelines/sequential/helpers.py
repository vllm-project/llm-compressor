import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from compressed_tensors import has_offloaded_params
from compressed_tensors.quantization import find_name_or_class_matches
from torch.fx import Graph, GraphModule, Node
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.utils.fx import HFTracer

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["trace_subgraphs", "Subgraph"]


@dataclass
class Subgraph:
    graph: Graph
    input_names: Set[str]
    consumed_names: Set[str]

    def compile_forward(self):
        code = self.graph.python_code("self")
        exec(code.src, code.globals)
        return code.globals.get("forward")


def trace_subgraphs(
    model: PreTrainedModel,
    sample_input: Dict[str, Any],
    sequential_targets: List[str],
    ignore: List[str],
) -> List[Subgraph]:
    # find modules
    sequential_targets = match_modules(model, sequential_targets)
    ignore = match_modules(model, ignore)

    # initialize arguments
    tracer = get_tracer(model, sequential_targets, ignore)
    concrete_args = populate_concrete_args(model, sample_input)

    # trace
    with (
        calibration_forward_context(model),
        HooksMixin.disable_hooks(),
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
    partitions = topological_partition(graph, sequential_targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    return subgraphs


def get_tracer(
    model: Module, sequential_targets: Set[Module], ignore: Set[Module]
) -> HFTracer:
    offloaded_modules = set(
        module for module in model.modules() if has_offloaded_params(module)
    )

    class PiecewiseTracer(HFTracer):
        # Treat as leaf, skip tracing inside this module
        def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
            return (
                module in sequential_targets
                or module in offloaded_modules
                or module in ignore
                or super().is_leaf_module(module, module_qualified_name)
            )

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

    # a perfect solution would involve implicitly consolodating partition indices so
    # that each node is assigned to the maximum partition possible (in order to delay
    # execution as long as possible), but this covers the most costly case (get_attr)
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

        assert check_assumption(graph)

    return subgraphs


def trace_consumed_names(subgraphs: List[Dict[str, Any]]):
    # populate consumed_names according to when inputs are last used
    # in order to vacate the `intermediates` cache and save memory
    all_input_names = set().union(*(subgraph.input_names for subgraph in subgraphs))
    for input_name in all_input_names:
        for subgraph in reversed(subgraphs):
            if input_name in subgraph.input_names:
                subgraph.consumed_names.add(input_name)
                break
        else:
            assert False


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


def match_modules(model: Module, target_names: List[str]) -> Set[Module]:
    return set(
        module
        for name, module in model.named_modules()
        if find_name_or_class_matches(name, module, target_names)
    )
