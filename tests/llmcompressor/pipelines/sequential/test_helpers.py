import math
import sys

import pytest
import torch
import torch.fx
from transformers import AutoModelForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.ast_helpers import autowrap_forward
from llmcompressor.pipelines.sequential.helpers import (
    Subgraph,
    find_target_nodes,
    get_sequential_ancestors,
    partition_graph,
    topological_partition,
    trace_consumed_names,
    trace_subgraphs,
)
from llmcompressor.utils.dev import skip_weights_download, skip_weights_initialize


def run_subgraphs(model, subgraphs, inputs):
    namespace = dict(inputs)
    for subgraph in subgraphs:
        subgraph_inputs = {name: namespace[name] for name in subgraph.input_names}
        output = subgraph.forward(model, **subgraph_inputs)
        if isinstance(output, dict):
            namespace.update(output)
        else:
            output_node = next(
                node for node in subgraph.graph.nodes if node.op == "output"
            )
            namespace[output_node.args[0].name] = output
    return namespace


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)


class DummyModelMultipleSequentialLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.layer4 = torch.nn.Linear(10, 10)
        self.layer5 = torch.nn.Linear(10, 10)
        self.layer6 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


def test_autowrap_forward_uses_unwrapped_function_globals(monkeypatch):
    def forward(self, x):
        return torch.relu(x)

    MainModule = type(
        "MainModule", (torch.nn.Module,), {"__module__": "__main__", "forward": forward}
    )
    model = MainModule()
    monkeypatch.delitem(sys.modules["__main__"].__dict__, "torch", raising=False)

    with autowrap_forward(model, ignore=[]):
        output = model(torch.ones(2))

    assert torch.equal(output, torch.ones(2))


class DummyModelWithBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.merge = torch.nn.Linear(20, 10)

    def forward(self, x):
        left = self.layer1(x)
        right = self.layer2(x)
        merged = torch.cat([left, right], dim=-1)
        merged = self.merge(merged)
        return self.layer3(merged)


def _multiple_layer_targets(model: DummyModelMultipleSequentialLayers):
    return {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }


def _assert_partition_coverage(graph_module, partitions, targets):
    all_partition_nodes = [node for partition in partitions for node in partition]
    graph_nodes = list(graph_module.graph.nodes)

    assert len(all_partition_nodes) == len(graph_nodes)
    assert len(all_partition_nodes) == len(set(all_partition_nodes))

    target_nodes = find_target_nodes(graph_module, targets)
    assert len(target_nodes) == len(targets)
    for target_node in target_nodes:
        assert sum(1 for partition in partitions if target_node in partition) == 1


def _assert_get_attr_precedes_consumers(partitions):
    for partition in partitions:
        node_index = {node: index for index, node in enumerate(partition)}
        for node in partition:
            if node.op != "get_attr":
                continue
            for user in node.users:
                if user in node_index:
                    assert node_index[node] < node_index[user]


def test_topological_partition_coverage():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    targets = _multiple_layer_targets(model)
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)

    _assert_partition_coverage(graph_module, partitions, targets)
    _assert_get_attr_precedes_consumers(partitions)


def test_partition_graph_forward_equivalence():
    model = DummyModelMultipleSequentialLayers()

    targets = _multiple_layer_targets(model)
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    sample_input = torch.randn(2, 10)
    expected = model(sample_input)
    namespace = run_subgraphs(model, subgraphs, {"x": sample_input})
    assert torch.allclose(namespace["layer6"], expected)


def test_partition_graph_branch_forward_equivalence():
    model = DummyModelWithBranch()

    targets = {model.layer1, model.layer2, model.layer3}
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    sample_input = torch.randn(2, 10)
    expected = model(sample_input)
    namespace = run_subgraphs(model, subgraphs, {"x": sample_input})
    assert torch.allclose(namespace["layer3"], expected)


def test_trace_consumed_names_last_use():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    targets = _multiple_layer_targets(model)
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    all_input_names = set().union(*(subgraph.input_names for subgraph in subgraphs))
    for input_name in all_input_names:
        consumers = [
            subgraph for subgraph in subgraphs if input_name in subgraph.consumed_names
        ]
        assert len(consumers) == 1

        last_subgraph = next(
            subgraph
            for subgraph in reversed(subgraphs)
            if input_name in subgraph.input_names
        )
        assert input_name in last_subgraph.consumed_names


def test_submodules_order_is_stable():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    targets = _multiple_layer_targets(model)
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)
    subgraphs = partition_graph(model, partitions)

    for subgraph in subgraphs:
        first = subgraph.submodules(model)
        second = subgraph.submodules(model)
        assert first == second


def test_get_sequential_ancestors():
    with skip_weights_initialize():
        model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}


def test_topological_partition_default():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }
    gm = torch.fx.symbolic_trace(model)

    assert len(topological_partition(gm, targets)) == 7


def test_topological_partition_multiple_targets():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    gm = torch.fx.symbolic_trace(model)
    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }

    assert len(topological_partition(gm, targets, 2)) == 4


def test_topological_partition_invalid():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    gm = torch.fx.symbolic_trace(model)
    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }

    with pytest.raises(ValueError):
        topological_partition(gm, targets, 0)


@pytest.mark.parametrize("targets_per_subgraph", [1, 2, 3, 4, 5])
def test_trace_subgraphs(targets_per_subgraph):
    target = "Qwen3DecoderLayer"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    subgraphs = trace_subgraphs(
        model,
        model.dummy_inputs,
        sequential_targets=[target],
        ignore=DatasetArguments().tracing_ignore,
        targets_per_subgraph=targets_per_subgraph,
    )

    # +1 refers to preamble before first target
    min_num_subgraphs = len(model.model.layers) // targets_per_subgraph + 1
    max_num_subgraphs = math.ceil(len(model.model.layers) / targets_per_subgraph) + 1
    assert min_num_subgraphs <= len(subgraphs) <= max_num_subgraphs
    for subgraph in subgraphs[1:-1]:  # only check middle, ends can can be non-divisible
        subgraph_modules = subgraph.submodules(model)
        num_targets_present = len(
            [
                module
                for module in subgraph_modules
                if module.__class__.__name__ == target
            ]
        )
        assert num_targets_present == targets_per_subgraph


@pytest.mark.parametrize(
    "input_names,expected_consumed_names",
    [
        ([], []),
        ([{"input"}], [{"input"}]),
        (
            [
                {"tokens", "mask"},
                {"hidden_0", "mask"},
                {"hidden_1", "mask"},
            ],
            [{"tokens"}, {"hidden_0"}, {"hidden_1", "mask"}],
        ),
        (
            [
                {"input", "skip"},
                {"hidden_0"},
                {"hidden_1", "skip"},
            ],
            [{"input"}, {"hidden_0"}, {"hidden_1", "skip"}],
        ),
    ],
)
def test_trace_consumed_names(input_names, expected_consumed_names):
    subgraphs = [
        Subgraph(
            graph=torch.fx.Graph(),
            input_names=names,
            consumed_names=set(),
        )
        for names in input_names
    ]
    original_input_names = [subgraph.input_names.copy() for subgraph in subgraphs]

    trace_consumed_names(subgraphs)

    assert [
        subgraph.consumed_names for subgraph in subgraphs
    ] == expected_consumed_names
    assert [subgraph.input_names for subgraph in subgraphs] == original_input_names
