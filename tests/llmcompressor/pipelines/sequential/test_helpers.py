import math
import sys

import pytest
import torch
import torch.fx
from transformers import AutoModelForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.ast_helpers import autowrap_forward
from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    SequentialTracer,
    Subgraph,
    collect_subgraph_modules,
    find_modules_outside_subgraphs,
    find_target_nodes,
    get_sequential_ancestors,
    handle_sequential_oom,
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


def test_submodules_includes_modules_invoked_inside_wrapped_nodes():
    """call_module-only discovery misses leaves executed inside fx.wrap (#3261)."""
    with skip_weights_initialize():
        model = DummyModel()

    graph = torch.fx.Graph(model)
    x = graph.placeholder("x")
    wrapped = graph.call_function(lambda t: t, (x,))
    wrapped.meta[INVOKED_SUBMODULES_META_KEY] = ["seq.0"]
    fc = graph.call_module("fc", (wrapped,))
    graph.output(fc)

    subgraph = Subgraph(graph=graph, input_names={"x"}, consumed_names=set())
    modules = subgraph.submodules(model)

    assert model.seq[0] in modules
    assert model.fc in modules
    assert model.seq[0] in subgraph.submodule_dict(model).values()


def test_create_proxy_recording_is_reentrant(monkeypatch):
    """Nested create_proxy must not wipe the parent's pending attributions."""

    class Leaf(torch.nn.Module):
        def forward(self, x):
            return x

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parent_mod = Leaf()
            self.inner_mod = Leaf()

        def forward(self, x):
            return x

    root = Root()
    tracer = SequentialTracer(ancestors=set(), targets=set())
    tracer._module_to_name = {
        module: name for name, module in root.named_modules() if name
    }
    handles = [
        root.parent_mod.register_forward_hook(tracer._record_invoked_module),
        root.inner_mod.register_forward_hook(tracer._record_invoked_module),
    ]
    inner_meta: dict[str, list[str]] = {}

    def fake_hf_create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        node = type("Node", (), {"meta": {}})()
        proxy = type("Proxy", (), {"node": node})()
        if kind == "outer":
            root.parent_mod(torch.zeros(1))
            inner = self.create_proxy("inner", None, (), {})
            inner_meta["names"] = list(
                inner.node.meta.get(INVOKED_SUBMODULES_META_KEY, [])
            )
            root.parent_mod(torch.zeros(1))
        else:
            root.inner_mod(torch.zeros(1))
        return proxy

    from llmcompressor.pipelines.sequential.transformers_helpers import HFTracer

    monkeypatch.setattr(HFTracer, "create_proxy", fake_hf_create_proxy)
    try:
        outer = tracer.create_proxy("outer", None, (), {})
    finally:
        for handle in handles:
            handle.remove()

    assert outer.node.meta[INVOKED_SUBMODULES_META_KEY] == ["parent_mod"]
    assert "inner_mod" not in outer.node.meta[INVOKED_SUBMODULES_META_KEY]
    assert inner_meta["names"] == ["inner_mod"]


def test_alias_module_call_recorded_during_create_proxy(monkeypatch):
    """Execution-based attribution must see `emb = self.embed; emb(x)` (#3261)."""

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 4)

        def forward(self, x):
            return x

    root = Root()
    tracer = SequentialTracer(ancestors=set(), targets=set())
    tracer._module_to_name = {
        module: name for name, module in root.named_modules() if name
    }
    handle = root.embed_tokens.register_forward_hook(tracer._record_invoked_module)

    def fake_hf_create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        emb = root.embed_tokens  # alias
        emb(torch.zeros(1, 3, dtype=torch.long))
        return type("Proxy", (), {"node": type("Node", (), {"meta": {}})()})()

    from llmcompressor.pipelines.sequential.transformers_helpers import HFTracer

    monkeypatch.setattr(HFTracer, "create_proxy", fake_hf_create_proxy)
    try:
        proxy = tracer.create_proxy("call_function", None, (), {})
    finally:
        handle.remove()

    assert proxy.node.meta[INVOKED_SUBMODULES_META_KEY] == ["embed_tokens"]
    assert "embed_tokens" in tracer.oracle_invoked_names


def test_unused_module_not_covered_by_subgraphs():
    with skip_weights_initialize():
        model = DummyModel()

    graph = torch.fx.Graph(model)
    x = graph.placeholder("x")
    fc = graph.call_module("fc", (x,))
    graph.output(fc)
    subgraph = Subgraph(graph=graph, input_names={"x"}, consumed_names=set())

    covered = set(subgraph.submodules(model))
    assert model.fc in covered
    assert model.seq[0] not in covered


def test_ancestor_not_owned_via_invoked_meta():
    with skip_weights_initialize():
        model = DummyModel()

    graph = torch.fx.Graph(model)
    x = graph.placeholder("x")
    # Even if meta incorrectly listed the root, ownership checks exclude ancestors
    # at the oracle layer; meta resolution still returns the module object, so ensure
    # subgraph helpers do not treat ancestors as the only coverage signal.
    wrapped = graph.call_function(lambda t: t, (x,))
    wrapped.meta[INVOKED_SUBMODULES_META_KEY] = ["fc"]
    graph.output(wrapped)
    subgraph = Subgraph(graph=graph, input_names={"x"}, consumed_names=set())

    names = subgraph._subgraph_module_names(model, recurse=False)
    assert "fc" in names
    assert "" not in names  # root / ancestor path never recorded as a leaf name


def test_trace_subgraphs_wrap_hidden_ownership_and_oracle():
    """
    Wrap-hidden embeddings must belong to subgraph 0 (offloading stages them there),
    remain absent as call_module, and satisfy oracle ⊆ covered (#3261).
    """
    model_id = "inference-optimization/Llama-3.2-0.5B-Instruct"
    with skip_weights_download(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    sample = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long, device="meta"),
        "attention_mask": torch.ones(1, 8, dtype=torch.long, device="meta"),
    }
    trace = trace_subgraphs(
        model,
        sample,
        sequential_targets=["LlamaDecoderLayer"],
        ignore=DatasetArguments().tracing_ignore,
    )
    subgraphs = trace.subgraphs

    # Correct subgraph: preamble owns embeddings; later subgraphs do not.
    assert "model.embed_tokens" in subgraphs[0].submodule_dict(model)
    for subgraph in subgraphs[1:]:
        assert "model.embed_tokens" not in subgraph.submodule_dict(model)

    assert not any(
        node.op == "call_module" and node.target == "model.embed_tokens"
        for subgraph in subgraphs
        for node in subgraph.graph.nodes
    )
    assert any(
        "model.embed_tokens" in node.meta.get(INVOKED_SUBMODULES_META_KEY, ())
        for node in subgraphs[0].graph.nodes
    )

    # Offloading: wrap-hidden leaf is subgraph-staged, not persistent/outside.
    outside = find_modules_outside_subgraphs(model, subgraphs)
    assert "model.embed_tokens" not in outside

    target_modules = {
        module
        for module in model.modules()
        if module.__class__.__name__ == "LlamaDecoderLayer"
    }
    ancestors = get_sequential_ancestors(model, target_modules)
    covered = collect_subgraph_modules(model, subgraphs)
    missing = sorted(
        name
        for name in trace.oracle_invoked_names
        if model.get_submodule(name) not in ancestors and name not in covered
    )
    assert missing == []
    assert "model.embed_tokens" in trace.oracle_invoked_names

    # Ancestors are not hooked; they must never appear in the oracle.
    ancestor_names = {
        name for name, module in model.named_modules() if module in ancestors
    }
    assert trace.oracle_invoked_names.isdisjoint(ancestor_names)


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
