
import contextlib
from typing import Any, Callable, Dict, List, Set

import torch
from collections import deque
from transformers import AutoModel
from torch.fx import GraphModule, Graph, Node
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.fx import symbolic_trace, HFTracer

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import EarlyStopException, apply_pad_mask_to_batch
from llmcompressor.pytorch.utils.helpers import tensors_to_device


def get_target_nodes(graph: GraphModule, targets: List[str]):
    target_nodes = []
    for node in graph.graph.nodes:
        if (
            node.op == "call_module" and
            type(graph.get_submodule(node.target)).__name__ in targets
        ):
            target_nodes.append(node)

    return target_nodes


def check_assumption(graph: Graph) -> bool:
    for node in graph.nodes:
        for user in node.users:
            if node not in user.all_input_nodes:
                return False

        for input_node in node.all_input_nodes:
            if node not in input_node.users:
                return False

        if (
            len(node.users) != len(set(node.users)) or 
            len(node.all_input_nodes) != len(set(node.all_input_nodes))
        ):
            return False

    return True


def topological_partition(graph: GraphModule, target_nodes: Set[Node]) -> List[List[Node]]:
    # use list representation to maintain topological sorting
    assert check_assumption(graph.graph)

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


def partition_graph(model: torch.nn.Module, partitions: List[List[Node]]):
    subgraphs = []

    # create subgraphs
    for partition_nodes in partitions:
        # create a new graph for the partition
        subgraph = Graph(model)
        node_map = {}

        # add placeholders for inputs not in this subgraph. use set to deduplicate
        new_input_nodes = {
            input_node
            for node in partition_nodes
            #if node.op != "get_attr"
            for input_node in node.all_input_nodes
            if input_node not in partition_nodes and input_node.op
        }
        for input_node in new_input_nodes:
            node_map[input_node] = subgraph.placeholder(input_node.name)

        # add the nodes to subgraph
        for node in partition_nodes:
            node_map[node] = subgraph.node_copy(node, lambda n: node_map[n])

        # add an output node to collect all subgraph outputs into a dictionary
        if len(subgraph.find_nodes(op="output")) <= 0:
            output_dict = {
                node.name: node_map[node]
                for node in partition_nodes
                if any(user not in partition_nodes for user in node.users.keys())
            }
            subgraph.output(output_dict)

        # Save the subgraph for this partition
        subgraph.lint()
        input_names = [node.name for node in subgraph.nodes if node.op == "placeholder"]
        subgraphs.append({
            "graph": subgraph,
            "code": subgraph.python_code("self"),
            "input_names": input_names,
            "consumed_names": [],
        })

        print([n for n in subgraph.nodes])
        assert check_assumption(subgraph)

    # populate consumed_names according to when inputs are last used
    # in order to vacate the `intermediates` cache and save memory
    all_input_names = set().union(*(subgraph["input_names"] for subgraph in subgraphs))
    for input_name in all_input_names:
        for subgraph in reversed(subgraphs):
            if input_name in subgraph["input_names"]:
                subgraph["consumed_names"].append(input_name)
                break
        else:
            assert False

    return subgraphs


class PartitionedModel:
    def __init__(self):
        self.hook_targets = []
        self.hook_target_nodes = []
        self.graph = None
        self.subgraphs = []
        self.model = None

    def register_hook(self, func: Callable, targets: List[str]):
        self.hook_targets.append((func, targets))

    def init_forward(self, model: torch.nn.Module, targets):
        self.model = model

        # 1. trace graph
        class CustomTracer(HFTracer):
            def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
                if type(module).__name__ in targets:
                    return True  # Treat as leaf, skip tracing inside this module
                return super().is_leaf_module(module, module_qualified_name)
        
        self.graph: GraphModule = symbolic_trace(model, disable_check=True, tracer_cls=CustomTracer)
        #self.graph: GraphModule = CustomTracer().trace(model, dummy_inputs=model.dummy_inputs)

        # 2. identify target nodes
        all_target_nodes = get_target_nodes(self.graph, targets)

        # 3. cut into partitions along target nodes
        partitions: List[List[Node]] = topological_partition(self.graph, all_target_nodes)
        self.subgraphs: List[GraphModule] = partition_graph(model, partitions)

    def forward_data(
        self,
        dataloader,
        mask_padding: bool = True,
        run_twice: bool = True
    ):
        # TODO: give option to skip lm_head
        # 4. perform compression
        model_device = next(self.model.parameters()).device
        batch_intermediates = [
            apply_pad_mask_to_batch(batch) if mask_padding else batch
            for batch in dataloader
        ]
        batch_outputs = [None for _ in range(len(dataloader))]

        for subgraph_index, subgraph in enumerate(self.subgraphs):
            code = subgraph["code"]
            exec(code.src, code.globals)
            forward_function = code.globals.get("forward")

            if run_twice:
                for batch_index in range(len(dataloader)):
                    intermediates = batch_intermediates[batch_index]
                    inputs = {input_name: intermediates[input_name] for input_name in subgraph["input_names"]}
                    inputs = tensors_to_device(inputs, model_device)
                    try:
                        forward_function(self.model, **inputs)
                    except EarlyStopException:
                        pass
                
            with HooksMixin.disable_hooks() if run_twice else contextlib.nullcontext():
                for batch_index in range(len(dataloader)):
                    intermediates = batch_intermediates[batch_index]

                    inputs = {input_name: intermediates[input_name] for input_name in subgraph["input_names"]}
                    inputs = tensors_to_device(inputs, model_device)
                    try:
                        subgraph_output = forward_function(self.model, **inputs)
                    except EarlyStopException:
                        subgraph_output = None
                        pass
                    subgraph_output = tensors_to_device(subgraph_output, "cpu")

                    for consumed_name in subgraph["consumed_names"]:
                        del intermediates[consumed_name]

                    if subgraph_index < len(self.subgraphs) - 1:
                        intermediates.update(subgraph_output)
                    else:
                        batch_outputs[batch_index] = subgraph_output

        return batch_outputs
