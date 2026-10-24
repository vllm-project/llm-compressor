import math
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.fx
from transformers import AutoModelForCausalLM, WhisperForConditionalGeneration

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.ast_helpers import autowrap_forward
from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    Subgraph,
    _resolve_sequential_targets,
    collect_subgraph_modules,
    find_modules_outside_graph,
    find_modules_outside_subgraphs,
    find_target_nodes,
    get_sequential_ancestors,
    handle_sequential_oom,
    partition_graph,
    partition_subgraphs,
    topological_partition,
    trace_consumed_names,
    trace_graph,
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


class DummyModelWithParameters(torch.nn.Module):
    """Reads parameters directly in forward, which produces `get_attr` nodes"""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(10, 10) for _ in range(6))
        self.scales = torch.nn.ParameterList(
            torch.nn.Parameter(torch.ones(10)) for _ in range(6)
        )

    def forward(self, x):
        for layer, scale in zip(self.layers, self.scales):
            x = layer(x * scale)
        return x


class DummyModelWithUnusedBranch(torch.nn.Module):
    """The side branch only runs for its effects, like a calibration hook"""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.side1 = torch.nn.Linear(10, 10)
        self.side2 = torch.nn.Linear(10, 10)
        self.head = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        self.side2(torch.relu(self.side1(x)))
        return self.head(x)


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


def test_partition_graph_unused_branch_forward_equivalence():
    model = DummyModelWithUnusedBranch()

    targets = {model.layer1, model.side1, model.side2}
    graph_module = torch.fx.symbolic_trace(model)
    partitions = topological_partition(graph_module, targets)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    # the output node is ready before side2, but must not end a partition early
    assert [node.op for node in partitions[-1]][-1] == "output"
    assert all(
        node.op != "output" for partition in partitions[:-1] for node in partition
    )

    sample_input = torch.randn(2, 10)
    expected = model(sample_input)
    namespace = run_subgraphs(model, subgraphs, {"x": sample_input})
    assert torch.allclose(namespace["head"], expected)


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


@pytest.mark.parametrize("targets_per_subgraph", [1, 2, 3, 4, 6])
def test_topological_partition_memory_matches_targets_per_subgraph(
    targets_per_subgraph,
):
    with skip_weights_initialize():
        model = DummyModelWithParameters()

    targets = set(model.layers)
    graph_module = torch.fx.symbolic_trace(model)
    assert len(graph_module.graph.find_nodes(op="get_attr")) > 0
    target_nodes = find_target_nodes(graph_module, targets)

    # when only targets cost memory, room for N targets partitions exactly like the
    # integer setting
    node_memory = {node: 1 for node in target_nodes}
    assert topological_partition(
        graph_module, targets, "auto", node_memory, targets_per_subgraph
    ) == topological_partition(graph_module, targets, targets_per_subgraph)


def test_topological_partition_memory_cuts_before_non_targets():
    model = DummyModelWithBranch()

    targets = {model.layer1, model.layer2, model.layer3}
    graph_module = torch.fx.symbolic_trace(model)

    # every module costs memory, so a cut lands before the non-target `merge`, which
    # then shares its partition with the next target
    node_memory = {node: 1 for node in graph_module.graph.find_nodes(op="call_module")}
    partitions = topological_partition(graph_module, targets, "auto", node_memory, 1)
    assert [[node.name for node in partition] for partition in partitions] == [
        ["x"],
        ["layer1"],
        ["layer2", "cat"],
        ["merge", "layer3", "output"],
    ]

    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)
    sample_input = torch.randn(2, 10)
    namespace = run_subgraphs(model, subgraphs, {"x": sample_input})
    assert torch.allclose(namespace["layer3"], model(sample_input))


def test_partition_subgraphs_auto_groups_whole_targets(monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.helpers.logger.warning", warning
    )
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    graph, targets = trace_graph(
        model,
        model.dummy_inputs,
        sequential_targets=["Qwen3DecoderLayer"],
        ignore=DatasetArguments().tracing_ignore,
    )
    num_layers = len(model.model.layers)
    target_nodes = find_target_nodes(graph, targets)

    # a plan with room for seven decoder layers per subgraph
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.helpers.plan_subgraph_memory",
        lambda *args: ({node: 1 for node in target_nodes}, 7),
    )
    auto_args = ([], [], DatasetArguments())  # dataloader, modifiers, dataset_args
    single = partition_subgraphs(model, graph, targets, 1)
    grouped = partition_subgraphs(model, graph, targets, "auto", *auto_args)

    # a group's modules are the union of its targets' modules, so per-target costs
    # add up to the cost of a group
    assert len(single) == num_layers + 1
    assert len(grouped) == num_layers // 7 + 1
    for index, subgraph in enumerate(grouped[1:]):
        expected = set().union(
            *(s.submodule_dict(model) for s in single[1 + 7 * index : 8 + 7 * index])
        )
        assert set(subgraph.submodule_dict(model)) == expected

    # without a plan, "auto" falls back to one target per subgraph
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.helpers.plan_subgraph_memory",
        lambda *args: None,
    )
    assert len(partition_subgraphs(model, graph, targets, "auto", *auto_args)) == len(
        single
    )

    # "auto" cannot estimate memory without the calibration data and modifiers
    with pytest.raises(ValueError, match="dataloader"):
        partition_subgraphs(model, graph, targets, "auto")

    # grouping more than one target per subgraph is not a target mismatch
    warning.assert_not_called()


def test_partition_subgraphs_warns_on_target_mismatch(monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.helpers.logger.warning", warning
    )
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    graph_module = torch.fx.symbolic_trace(model)
    targets = _multiple_layer_targets(model)
    partition_subgraphs(model, graph_module, targets, 4)
    warning.assert_not_called()

    uncalled = torch.nn.Linear(10, 10)
    partition_subgraphs(model, graph_module, targets | {uncalled}, 4)
    warning.assert_called_once()


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


def test_handle_sequential_oom_mentions_calibration_levers():
    @handle_sequential_oom
    def pipeline():
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.OutOfMemoryError) as excinfo:
        pipeline()

    message = str(excinfo.value)
    assert "Sequential pipeline ran out of memory" in message
    assert "`max_seq_length`" in message
    assert "`num_calibration_samples`" in message


@pytest.mark.parametrize("targets_per_subgraph,suggests_auto", [(1, False), (4, True)])
def test_handle_sequential_oom_suggests_auto_only_with_several_targets(
    targets_per_subgraph, suggests_auto
):
    @handle_sequential_oom
    def pipeline(model, dataloader, dataset_args):
        raise torch.OutOfMemoryError("CUDA out of memory")

    dataset_args = DatasetArguments(
        sequential_targets_per_subgraph=targets_per_subgraph
    )
    with pytest.raises(torch.OutOfMemoryError) as excinfo:
        pipeline(None, None, dataset_args)

    # "auto" never puts fewer than one target in a subgraph, so it cannot help at 1
    message = str(excinfo.value)
    assert ("sequential_targets_per_subgraph='auto' chooses" in message) == (
        suggests_auto
    )


def test_handle_sequential_oom_with_auto_suggests_more_headroom():
    @handle_sequential_oom
    def pipeline(model, dataloader, dataset_args):
        raise torch.OutOfMemoryError("CUDA out of memory")

    dataset_args = DatasetArguments(sequential_targets_per_subgraph="auto")
    with pytest.raises(torch.OutOfMemoryError) as excinfo:
        pipeline(None, None, dataset_args)

    # a run which already uses "auto" is told how to leave more headroom instead
    message = str(excinfo.value)
    assert "`gpu_memory_utilization`" in message
    assert "'auto' chooses" not in message


# --- modules invoked inside wrapped functions (#3261) ---


def _subgraph_with_invoked_meta(model, invoked_names):
    """Build a one-node subgraph whose `call_function` node records `invoked_names`."""
    graph = torch.fx.Graph(model)
    x = graph.placeholder("x")
    wrapped = graph.call_function(lambda t: t, (x,))
    wrapped.meta[INVOKED_SUBMODULES_META_KEY] = list(invoked_names)
    graph.output(wrapped)
    return Subgraph(graph=graph, input_names={"x"}, consumed_names=set())


def _trace_meta_model(model, sample_input, sequential_targets, **kwargs):
    """Trace `model` on meta tensors with the default tracing ignore list."""
    return trace_subgraphs(
        model,
        sample_input,
        sequential_targets=sequential_targets,
        ignore=DatasetArguments().tracing_ignore,
        **kwargs,
    )


@pytest.fixture(scope="module")
def llama_meta_model():
    """Llama on the meta device, shared by the wrap-hidden ownership tests."""
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            "inference-optimization/Llama-3.2-0.5B-Instruct", device_map="meta"
        )
    sample_input = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long, device="meta"),
        "attention_mask": torch.ones(1, 8, dtype=torch.long, device="meta"),
    }
    return model, sample_input


def test_submodules_includes_modules_invoked_inside_wrapped_nodes():
    with skip_weights_initialize():
        model = DummyModel()

    subgraph = _subgraph_with_invoked_meta(model, ["seq"])

    assert subgraph.submodules(model) == [model.seq, model.seq[0], model.seq[1]]
    assert subgraph.submodules(model, recurse=False) == [model.seq]
    assert model.fc not in subgraph.submodules(model)


def test_subgraph_module_names_fail_loud_on_unknown_name():
    with skip_weights_initialize():
        model = DummyModel()

    subgraph = _subgraph_with_invoked_meta(model, ["does.not.exist"])

    with pytest.raises(AttributeError):
        subgraph.submodules(model)


@pytest.mark.parametrize("targets_per_subgraph", [1, 2])
def test_trace_subgraphs_wrap_hidden_ownership(llama_meta_model, targets_per_subgraph):
    # `embed_tokens` runs inside the autowrapped `LlamaModel.forward`, so it is never a
    # `call_module` node and must be attributed to the preamble subgraph instead
    model, sample_input = llama_meta_model
    embed_tokens = model.model.embed_tokens
    root_children = list(model._modules)

    subgraphs = _trace_meta_model(
        model,
        sample_input,
        ["LlamaDecoderLayer"],
        targets_per_subgraph=targets_per_subgraph,
    )

    owners = [
        index
        for index, subgraph in enumerate(subgraphs)
        if embed_tokens in subgraph.submodules(model)
    ]
    assert owners == [0]
    assert "model.embed_tokens" not in find_modules_outside_subgraphs(model, subgraphs)
    # recording must not insert modules constructed in `forward` into the model
    assert list(model._modules) == root_children


def test_call_module_nodes_carry_no_invoked_meta(llama_meta_model):
    # `call_module` nodes already own their descendants through `recurse`, so the
    # tracer must only annotate nodes whose module calls are otherwise invisible
    model, sample_input = llama_meta_model
    subgraphs = _trace_meta_model(model, sample_input, ["LlamaDecoderLayer"])

    annotated = [
        node
        for subgraph in subgraphs
        for node in subgraph.graph.nodes
        if INVOKED_SUBMODULES_META_KEY in node.meta
    ]
    assert annotated, "expected at least one wrap-hidden module to be recorded"
    assert all(node.op == "call_function" for node in annotated)
    assert ["model.embed_tokens"] in [
        node.meta[INVOKED_SUBMODULES_META_KEY] for node in annotated
    ]


def test_whisper_encoder_covered():
    # The entire Whisper encoder runs inside a wrapped function. Every module which
    # is not an ancestor of a sequential target must still belong to some subgraph
    with skip_weights_download(WhisperForConditionalGeneration):
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny", device_map="meta"
        )
    sample_input = {
        "input_features": torch.zeros(1, 80, 3000, device="meta", dtype=model.dtype),
        "decoder_input_ids": torch.zeros(1, 4, dtype=torch.long, device="meta"),
    }
    sequential_targets = ["WhisperDecoderLayer"]

    subgraphs = _trace_meta_model(model, sample_input, sequential_targets)

    targets = _resolve_sequential_targets(model, sequential_targets)
    ancestors = get_sequential_ancestors(model, targets)
    outside = find_modules_outside_subgraphs(model, subgraphs)
    assert [name for name, module in outside.items() if module not in ancestors] == []

    encoder_names = {
        f"model.encoder.{name}"
        for name, _ in model.model.encoder.named_modules()
        if name
    }
    assert encoder_names <= set(collect_subgraph_modules(model, subgraphs))
    assert encoder_names <= set(subgraphs[0].submodule_dict(model))


@pytest.mark.parametrize("targets_per_subgraph", [1, "auto"])
def test_whole_model_target_is_one_untraced_subgraph(targets_per_subgraph):
    """The root as the only target, as in the basic pipeline, runs untraced"""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
    )
    sample_input = {"input": torch.ones(1, 4)}

    graph, targets = trace_graph(model, sample_input, [""], ignore=[])
    assert targets == {model}
    assert find_modules_outside_graph(model, graph) == {}

    subgraphs = partition_subgraphs(model, graph, targets, targets_per_subgraph)
    assert len(subgraphs) == 1
    assert subgraphs[0].input_names == {"input"}
    assert {"0", "1", "2"} <= set(subgraphs[0].submodule_dict(model))
    assert torch.equal(
        subgraphs[0].forward(model, **sample_input), model(**sample_input)
    )
