from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
import torch.fx
from compressed_tensors.offload import offload_module
from datasets import Dataset
from loguru import logger
from tokenizers import Tokenizer, models
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import active_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.pipelines.sequential import grouping, helpers
from llmcompressor.pipelines.sequential.grouping import (
    ACTIVATION_BYTES,
    estimate_activation_memory,
    get_input_shapes,
    get_memory_budget,
    get_module_costs,
    get_onload_bytes,
    plan_subgraph_memory,
)
from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    find_target_nodes,
    get_node_modules,
    topological_partition,
)
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.utils.dev import get_main_device
from tests.testing_utils import requires_gpu

_PIPELINE = "llmcompressor.pipelines.sequential.pipeline"


class FourLayers(torch.nn.Module):
    """Four target layers, then a head after the last target (like an lm_head)"""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(4))
        self.head = torch.nn.Linear(4, 4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _group_sizes(module_memory: dict[str, int], max_memory: int) -> list[int]:
    """
    Partition FourLayers by memory, where each node costs the memory of the modules
    it calls, and count the targets in each partition after the first
    """
    model = FourLayers()
    targets = set(model.layers)
    graph_module = torch.fx.symbolic_trace(model)
    node_memory = {
        node: sum(module_memory.get(name, 0) for name in get_node_modules(model, node))
        for node in graph_module.graph.nodes
    }
    partitions = topological_partition(
        graph_module, targets, "auto", node_memory, max_memory
    )
    target_nodes = find_target_nodes(graph_module, targets)
    return [len(target_nodes.intersection(p)) for p in partitions][1:]


LAYERS = {f"layers.{index}": 10 for index in range(4)}


def test_partition_by_memory_fits_all():
    assert _group_sizes(LAYERS | {"head": 10}, max_memory=100) == [4]


def test_partition_by_memory_splits_at_budget():
    # three layers fit in 30; the last layer starts a new partition with the head
    assert _group_sizes(LAYERS | {"head": 10}, max_memory=30) == [3, 1]


def test_partition_by_memory_cuts_before_costly_nodes_after_targets():
    # the head does not fit with the last two layers, so it gets its own partition
    assert _group_sizes(LAYERS | {"head": 10}, max_memory=25) == [2, 2, 0]


def test_partition_by_memory_oversized_target_gets_its_own_partition():
    assert _group_sizes(LAYERS | {"layers.1": 100}, max_memory=50) == [1, 1, 2]


def test_partition_by_memory_keeps_free_nodes_with_the_node_before_them():
    # the head onloads nothing, so it stays with the oversized last layer rather
    # than starting a partition of its own
    assert _group_sizes(LAYERS | {"layers.3": 100}, max_memory=50) == [3, 1]


def test_partition_by_memory_requires_node_memory():
    model = FourLayers()
    with pytest.raises(ValueError, match="node_memory"):
        topological_partition(torch.fx.symbolic_trace(model), set(model.layers), "auto")


class NormsBetweenLayers(torch.nn.Module):
    """A norm which onloads weights before every target layer"""

    def __init__(self):
        super().__init__()
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(4) for _ in range(4))
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(4))

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = layer(norm(x))
        return x


def test_partition_by_memory_never_splits_more_than_one_target_each():
    model = NormsBetweenLayers()
    targets = set(model.layers)
    graph_module = torch.fx.symbolic_trace(model)
    target_nodes = find_target_nodes(graph_module, targets)
    node_memory = {
        node: 10 if node in target_nodes else 1
        for node in graph_module.graph.find_nodes(op="call_module")
    }

    # each target fills the budget, so every norm would otherwise get its own
    # partition. Instead, each norm joins the partition of the target after it
    partitions = topological_partition(
        graph_module, targets, "auto", node_memory, max_memory=10
    )
    assert len(partitions) == len(topological_partition(graph_module, targets, 1))
    assert all(len(target_nodes.intersection(p)) == 1 for p in partitions[1:])


def test_get_node_modules():
    model = FourLayers()
    graph_module = torch.fx.symbolic_trace(model)
    node = next(n for n in graph_module.graph.nodes if n.target == "layers.0")
    assert get_node_modules(model, node) == ["layers.0"]

    # modules run inside wrapped code are recorded on the node by the tracer (#3264)
    node.meta[INVOKED_SUBMODULES_META_KEY] = ["layers"]
    assert get_node_modules(model, node) == [
        "layers.0",
        "layers",
        "layers.0",
        "layers.1",
        "layers.2",
        "layers.3",
    ]


class _KnownModifier:
    """Holds a byte per calibration token for every module, and 20 bytes of scratch"""

    def estimate_sequential_memory(self, module, num_tokens):
        return num_tokens, 20


class _UnknownModifier:
    def estimate_sequential_memory(self, module, num_tokens):
        return None


def test_get_module_costs_reports_unknown_modifiers():
    model = FourLayers()
    modifiers = [_KnownModifier(), _UnknownModifier()]
    costs, unknown = get_module_costs(model, {"head"}, modifiers, num_tokens=7)
    assert costs == {"head": (7, 20)}  # not offloaded, so onloading adds nothing
    assert unknown == {"_UnknownModifier"}

    _, unknown = get_module_costs(model, {"head"}, [_KnownModifier()], num_tokens=7)
    assert unknown == set()


def test_get_onload_bytes():
    linear = torch.nn.Linear(16, 8)
    assert get_onload_bytes(linear) == 0  # not offloaded

    offload_module(linear, onload_device="cpu", offload_device="cpu")
    assert get_onload_bytes(linear) == 0  # onloading does not add memory


@requires_gpu
def test_get_onload_bytes_offloaded():
    linear = torch.nn.Linear(16, 8, dtype=torch.bfloat16)
    linear.register_buffer("scale", torch.ones(8))
    offload_module(linear, onload_device=get_main_device(), offload_device="cpu")

    allocated = torch.accelerator.memory_allocated()
    assert get_onload_bytes(linear) == (16 * 8 + 8) * 2 + 8 * 4
    assert torch.accelerator.memory_allocated() == allocated  # nothing was onloaded


def _batches(*shapes):
    return [{"input_ids": torch.zeros(shape, dtype=torch.long)} for shape in shapes]


def test_get_input_shapes():
    # largest batch, longest sequence, and every token including padding
    assert get_input_shapes(_batches((2, 7), (3, 5))) == (3, 7, 2 * 7 + 3 * 5)

    assert get_input_shapes([{"inputs_embeds": torch.zeros(1, 4, 16)}]) is None
    assert get_input_shapes(_batches((7,))) is None
    assert get_input_shapes([(torch.zeros(1, 4),)]) is None


def test_get_input_shapes_reads_the_batches_calibrated_on():
    samples = [
        {"input_ids": torch.zeros(1, length, dtype=torch.long)}
        for length in range(1, 9)
    ]
    sampler = RandomSampler(samples, num_samples=4)
    dataloader = DataLoader(samples, sampler=sampler, collate_fn=lambda b: b[0])

    # a shuffling sampler picks the same random batches on the pass after counting
    torch.manual_seed(0)
    shapes = get_input_shapes(dataloader)
    lengths = [batch["input_ids"].shape[1] for batch in dataloader]
    assert shapes == (1, max(lengths), sum(lengths))


def test_estimate_activation_memory():
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(hidden_size=16)

    expected = 3 * 7 * 16 * ACTIVATION_BYTES
    assert estimate_activation_memory(model, (3, 7, 29)) == expected
    assert estimate_activation_memory(model, None) is None

    model.config = SimpleNamespace()
    assert estimate_activation_memory(model, (3, 7, 29)) is None


def test_get_memory_budget(monkeypatch):
    GiB = 1024**3
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device: (30 * GiB, 40 * GiB)
    )
    # 90% of 40 GiB may be used, of which 10 GiB already is
    assert get_memory_budget(torch.device("cuda"), 0.9) == pytest.approx(26 * GiB)
    assert get_memory_budget(torch.device("cuda"), 1.0) == pytest.approx(30 * GiB)

    def unsupported(device):
        raise RuntimeError("not supported")

    monkeypatch.setattr(torch.accelerator, "get_memory_info", unsupported)
    assert get_memory_budget(torch.device("cuda"), 0.9) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequential_targets_per_subgraph": "max"},
        {"sequential_targets_per_subgraph": 0},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 90},
    ],
)
def test_dataset_arguments_reject_invalid_grouping(kwargs):
    with pytest.raises(ValueError):
        DatasetArguments(**kwargs)


def test_dataset_arguments_accept_grouping():
    DatasetArguments(sequential_targets_per_subgraph="auto", gpu_memory_utilization=1)
    DatasetArguments(sequential_targets_per_subgraph=4)


def _plan(monkeypatch, budget, modifiers=(), batches=None, dataset_args=None):
    """Plan FourLayers on a simulated accelerator with the given budget"""
    monkeypatch.setattr(grouping, "get_main_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(grouping, "get_memory_budget", lambda *args: budget)

    model = FourLayers()
    model.config = SimpleNamespace(hidden_size=4)
    graph_module = torch.fx.symbolic_trace(model)
    node_modules = {
        node: get_node_modules(model, node) for node in graph_module.graph.nodes
    }
    plan = plan_subgraph_memory(
        model=model,
        node_modules=node_modules,
        dataloader=_batches((1, 8)) if batches is None else batches,
        modifiers=list(modifiers),
        dataset_args=DatasetArguments() if dataset_args is None else dataset_args,
    )
    return plan, {node.target: node for node in graph_module.graph.nodes}


def test_plan_subgraph_memory(monkeypatch):
    activation_memory = 1 * 8 * 4 * ACTIVATION_BYTES
    plan, nodes = _plan(monkeypatch, budget=10_000, modifiers=[_KnownModifier()])
    node_memory, max_memory = plan

    # every module holds a byte per token (8), and the scratch is smaller than the
    # activations, which are reserved from the budget
    assert node_memory[nodes["layers.0"]] == 8
    assert node_memory[nodes["head"]] == 8
    assert max_memory == 10_000 - activation_memory


def test_plan_subgraph_memory_without_accelerator(monkeypatch):
    monkeypatch.setattr(grouping, "get_main_device", lambda: torch.device("cpu"))
    assert plan_subgraph_memory(FourLayers(), {}, [], [], DatasetArguments()) is None


@pytest.mark.parametrize(
    "offload_device,log_sequential_error,copies",
    [("cpu", False, 0), ("cuda", False, 1), ("cuda:0", True, 2), ("cuda:1", False, 0)],
)
def test_plan_subgraph_memory_reserves_activations_cached_on_device(
    monkeypatch, offload_device, log_sequential_error, copies
):
    dataset_args = DatasetArguments(
        sequential_offload_device=offload_device,
        log_sequential_error=log_sequential_error,
    )
    plan, _ = _plan(monkeypatch, budget=10_000, dataset_args=dataset_args)
    _, max_memory = plan

    # hidden states of every token (1 * 8) at 4 float32 hidden units, per copy
    activation_memory = 1 * 8 * 4 * ACTIVATION_BYTES
    assert max_memory == 10_000 - activation_memory - copies * 8 * 4 * 4


@pytest.mark.parametrize("budget", [None, 0, 1 * 8 * 4 * ACTIVATION_BYTES])
def test_plan_subgraph_memory_falls_back_without_budget(monkeypatch, budget):
    plan, _ = _plan(monkeypatch, budget=budget)
    assert plan is None


def test_plan_subgraph_memory_falls_back_for_unknown_memory(monkeypatch):
    plan, _ = _plan(monkeypatch, budget=10_000, modifiers=[_UnknownModifier()])
    assert plan is None

    plan, _ = _plan(monkeypatch, budget=10_000, batches=[{"inputs_embeds": None}])
    assert plan is None


def _reduce(monkeypatch, local, other):
    """Reduce (budget, shapes) across this rank and one other rank"""

    def encode(budget, shapes):
        values = [budget is not None, shapes is not None, budget or 0]
        return torch.tensor(values + [-value for value in shapes or (0, 0, 0)])

    def all_reduce(values, op):
        assert op == grouping.dist.ReduceOp.MIN
        values.copy_(torch.minimum(values, encode(*other)))

    monkeypatch.setattr(grouping.dist, "all_reduce", all_reduce)
    return grouping.reduce_across_ranks(*local, torch.device("cpu"))


def test_reduce_across_ranks(monkeypatch):
    # the smallest budget, and the largest batch, sequence and number of tokens
    reduced = _reduce(monkeypatch, (1000, (2, 16, 64)), (900, (4, 8, 32)))
    assert reduced == (900, (4, 16, 64))


def test_reduce_across_ranks_falls_back_together(monkeypatch):
    # a value which one rank could not measure is unknown on every rank
    assert _reduce(monkeypatch, (1000, (2, 16, 64)), (None, (2, 16, 64))) == (
        None,
        (2, 16, 64),
    )
    assert _reduce(monkeypatch, (1000, (2, 16, 64)), (900, None)) == (900, None)


@pytest.fixture()
def fake_pipeline_inputs():
    model = torch.nn.Linear(4, 4)
    session = active_session()
    session.reset()
    session.initialize(model=model, start=-1)

    samples = [{"h": torch.ones(1, 4)} for _ in range(2)]
    dataloader = DataLoader(samples, batch_size=1, collate_fn=lambda b: b[0])

    yield model, dataloader

    session.finalize()


class _FakeSubgraph:
    """Passes its input through, and owns no modules"""

    input_names = {"h"}
    consumed_names = set()

    def forward(self, model, **kwargs):
        return {"h": kwargs["h"]}

    def submodules(self, model, recurse=True):
        return []

    def submodule_dict(self, model, recurse=True):
        return {}


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
@patch(f"{_PIPELINE}.partition_subgraphs")
@patch(f"{_PIPELINE}.find_modules_outside_graph", return_value={})
@patch(f"{_PIPELINE}.trace_graph")
def test_pipeline_traces_then_partitions(
    mock_trace, mock_outside, mock_partition, mock_targets, fake_pipeline_inputs
):
    graph, targets = MagicMock(), {MagicMock()}
    mock_trace.return_value = (graph, targets)
    mock_partition.return_value = [_FakeSubgraph()]
    model, dataloader = fake_pipeline_inputs

    dataset_args = DatasetArguments(sequential_targets_per_subgraph="auto")
    SequentialPipeline()(model, dataloader, dataset_args)

    # the graph is traced once and partitioned once, with what "auto" needs
    mock_trace.assert_called_once()
    mock_partition.assert_called_once_with(
        model, graph, targets, "auto", dataloader, ANY, dataset_args
    )


def _run_oneshot(tmp_path, recipe, targets_per_subgraph="auto"):
    """
    Run oneshot on a tiny seeded model without network access, and return the logged
    messages and the calibrated weights
    """
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,  # multiples of the W4A16 group size
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path)
    num_samples, seq_len = 4, 16
    dataset = Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 128, (num_samples, seq_len)).tolist(),
            "attention_mask": [[1] * seq_len] * num_samples,
        }
    )
    # only needed to satisfy oneshot, the dataset is already tokenized
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )

    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=num_samples,
            sequential_targets_per_subgraph=targets_per_subgraph,
        )
    finally:
        logger.remove(sink)

    weights = {name: value.clone() for name, value in model.state_dict().items()}
    return messages, weights


parametrize_auto_recipes = pytest.mark.parametrize(
    "recipe",
    [
        lambda: GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        lambda: [
            AWQModifier(),
            QuantizationModifier(
                targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"]
            ),
        ],
    ],
    ids=["gptq", "awq"],
)


@requires_gpu
@parametrize_auto_recipes
def test_oneshot_auto_group_sizes(tmp_path, recipe):
    messages, _ = _run_oneshot(tmp_path, recipe())

    # a tiny model fits in a single subgraph
    assert any("subgraphs of sizes [4]" in message for message in messages)


@parametrize_auto_recipes
def test_oneshot_auto_matches_fixed_grouping(tmp_path, monkeypatch, recipe):
    """
    "auto" end to end on any device: with the real memory estimates and room for two
    decoder layers per subgraph, it calibrates exactly like two targets per subgraph
    """
    monkeypatch.setattr(grouping, "get_main_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(grouping, "get_memory_budget", lambda *args: 1024**3)
    plan = helpers.plan_subgraph_memory

    def room_for_two_layers(*args):
        node_memory, _ = plan(*args)
        layer = max(
            memory
            for node, memory in node_memory.items()
            if node.op == "call_module" and node.target.startswith("model.layers.")
        )
        return node_memory, 2 * layer + layer // 2

    monkeypatch.setattr(helpers, "plan_subgraph_memory", room_for_two_layers)
    messages, auto_weights = _run_oneshot(tmp_path / "auto", recipe())
    _, fixed_weights = _run_oneshot(tmp_path / "fixed", recipe(), 2)

    assert any("subgraphs of sizes [2, 2]" in message for message in messages)
    assert auto_weights.keys() == fixed_weights.keys()
    for name, weight in auto_weights.items():
        assert torch.equal(weight, fixed_weights[name]), name


@requires_gpu
def test_oneshot_auto_falls_back_for_unknown_memory(tmp_path):
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
    messages, _ = _run_oneshot(tmp_path, recipe)

    # SmoothQuant does not estimate its memory, which may grow with the number of
    # targets per subgraph
    assert any(
        "SmoothQuantModifier" in message and "Falling back" in message
        for message in messages
    )
    assert not any("subgraphs of sizes" in message for message in messages)
