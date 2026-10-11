from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.fx
from compressed_tensors.offload import offload_module
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import active_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential import grouping
from llmcompressor.pipelines.sequential.grouping import (
    ACTIVATION_BYTES,
    SubgraphMemory,
    count_calibration_tokens,
    estimate_activation_memory,
    get_memory_budget,
    get_module_costs,
    get_node_modules,
    get_onload_bytes,
    plan_subgraph_memory,
)
from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    find_target_nodes,
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


class SharedModule(torch.nn.Module):
    """Every target is followed by the same shared module"""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(4))
        self.shared = torch.nn.Linear(4, 4)

    def forward(self, x):
        for layer in self.layers:
            x = self.shared(layer(x))
        return x


def _group_sizes(model, costs, activation_memory, budget):
    """Partition with a memory check and count the targets in each partition"""
    targets = set(model.layers)
    graph_module = torch.fx.symbolic_trace(model)
    memory = SubgraphMemory(model, costs, activation_memory, budget)
    partitions = topological_partition(graph_module, targets, is_full=memory)
    target_nodes = find_target_nodes(graph_module, targets)
    return [len(target_nodes.intersection(p)) for p in partitions][1:]


def _costs(**costs):
    """Module costs as (weight, held, scratch) bytes, keyed by module name"""
    return {name.replace("_", "."): tuple(cost) for name, cost in costs.items()}


LAYER_COSTS = _costs(
    layers_0=(10, 0, 0), layers_1=(10, 0, 0), layers_2=(10, 0, 0), layers_3=(10, 0, 0)
)


def test_subgraph_memory_fits_all():
    costs = {**LAYER_COSTS, "head": (10, 0, 0)}
    assert _group_sizes(FourLayers(), costs, activation_memory=5, budget=100) == [4]


def test_subgraph_memory_splits_at_budget():
    costs = {**LAYER_COSTS, "head": (10, 0, 0)}
    # three layers + activations = 35 fit; the last layer and the head start a group
    assert _group_sizes(FourLayers(), costs, activation_memory=5, budget=35) == [3, 1]


def test_subgraph_memory_counts_modules_after_the_last_target():
    costs = {**LAYER_COSTS, "head": (10, 0, 0)}
    # the fourth layer alone would fit with the third (25), but not together with
    # the head that runs after it (35)
    assert _group_sizes(FourLayers(), costs, activation_memory=5, budget=30) == [
        2,
        1,
        1,
    ]


def test_subgraph_memory_counts_held_memory():
    costs = {name: (10, 10, 0) for name in LAYER_COSTS} | {"head": (0, 0, 0)}
    assert _group_sizes(FourLayers(), costs, activation_memory=0, budget=45) == [2, 2]


def test_subgraph_memory_scratch_and_activations_do_not_add():
    costs = {name: (10, 0, 8) for name in LAYER_COSTS} | {"head": (0, 0, 0)}
    # 40 + max(10, 8) = 50 fits; 40 + 10 + 8 = 58 would not
    assert _group_sizes(FourLayers(), costs, activation_memory=10, budget=50) == [4]
    assert _group_sizes(FourLayers(), costs, activation_memory=10, budget=49) == [3, 1]


def test_subgraph_memory_counts_shared_module_once():
    costs = {**LAYER_COSTS, "shared": (20, 0, 0)}
    # two layers + the shared module = 40 fit; counted per call it would be 60
    assert _group_sizes(SharedModule(), costs, activation_memory=0, budget=45) == [2, 2]


def test_subgraph_memory_oversized_target_gets_its_own_group():
    costs = {**LAYER_COSTS, "head": (0, 0, 0), "layers.1": (100, 0, 0)}
    assert _group_sizes(FourLayers(), costs, activation_memory=0, budget=50) == [
        1,
        1,
        2,
    ]


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


def test_get_module_costs_reports_unknown_modifiers():
    class KnownModifier:
        def estimate_sequential_memory(self, module):
            return 10, 20

    class UnknownModifier:
        def estimate_sequential_memory(self, module):
            return None

    model = FourLayers()
    costs, unknown = get_module_costs(
        model, {"head"}, [KnownModifier(), UnknownModifier()]
    )
    assert costs == {"head": (0, 10, 20)}  # not offloaded, so onloading adds nothing
    assert unknown == {"UnknownModifier"}

    _, unknown = get_module_costs(model, {"head"}, [KnownModifier()])
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


def _cache(*shapes):
    batches = [{"input_ids": torch.zeros(shape, dtype=torch.long)} for shape in shapes]
    return IntermediatesCache.from_dataloader(batches, torch.device("cpu"))


def test_estimate_activation_memory():
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(hidden_size=16)

    # 3 samples (largest batch) * 7 tokens (longest batch) * 16 hidden
    activations = _cache((2, 7), (3, 5))
    expected = 3 * 7 * 16 * ACTIVATION_BYTES
    assert estimate_activation_memory(model, activations) == expected

    model.config = SimpleNamespace()
    assert estimate_activation_memory(model, activations) is None

    model.config = SimpleNamespace(hidden_size=16)
    no_ids = IntermediatesCache.from_dataloader(
        [{"inputs_embeds": torch.zeros(1, 4, 16)}], torch.device("cpu")
    )
    assert estimate_activation_memory(model, no_ids) is None


def test_count_calibration_tokens():
    assert count_calibration_tokens(_cache((2, 7), (3, 5))) == 2 * 7 + 3 * 5

    no_ids = IntermediatesCache.from_dataloader(
        [{"inputs_embeds": torch.zeros(1, 4, 16)}], torch.device("cpu")
    )
    assert count_calibration_tokens(no_ids) is None


def test_get_memory_budget(monkeypatch):
    GiB = 1024**3
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device: (30 * GiB, 40 * GiB)
    )
    # 90% of 40 GiB may be used, of which 10 GiB already is
    assert get_memory_budget(torch.device("cuda"), 0.9) == pytest.approx(26 * GiB)
    assert get_memory_budget(torch.device("cuda"), 1.0) == pytest.approx(30 * GiB)

    with pytest.raises(ValueError):
        get_memory_budget(torch.device("cuda"), 0)
    with pytest.raises(ValueError):
        get_memory_budget(torch.device("cuda"), 1.5)

    def unsupported(device):
        raise RuntimeError("not supported")

    monkeypatch.setattr(torch.accelerator, "get_memory_info", unsupported)
    assert get_memory_budget(torch.device("cuda"), 0.9) is None


def test_plan_subgraph_memory_without_accelerator():
    memory = plan_subgraph_memory(
        model=torch.nn.Linear(4, 4),
        graph=MagicMock(),
        activations=MagicMock(),
        modifiers=[],
        device=torch.device("cpu"),
        gpu_memory_utilization=0.9,
    )
    assert memory is None


def _plan_four_layers():
    model = FourLayers()
    model.config = SimpleNamespace(hidden_size=4)
    return plan_subgraph_memory(
        model=model,
        graph=torch.fx.symbolic_trace(model),
        activations=_cache((1, 8)),
        modifiers=[],
        device=torch.device("cuda"),
        gpu_memory_utilization=0.9,
    )


@pytest.mark.parametrize("budget,planned", [(1000, True), (0, False), (-100, False)])
def test_plan_subgraph_memory_falls_back_without_budget(monkeypatch, budget, planned):
    monkeypatch.setattr(grouping, "get_memory_budget", lambda *args: budget)
    memory = _plan_four_layers()
    assert (memory is not None) == planned


def test_reduce_across_ranks(monkeypatch):
    """Ranks agree on the smallest budget and the largest estimates"""

    def all_reduce(values, op):
        assert op == grouping.dist.ReduceOp.MIN
        # the other rank has 1 byte less budget and 1 byte more of every estimate
        values.copy_(torch.minimum(values, values - 1))

    monkeypatch.setattr(grouping.dist, "all_reduce", all_reduce)

    costs = _costs(layers_0=(10, 20, 30), head=(0, 5, 0))
    budget, activation_memory, costs = grouping.reduce_across_ranks(
        1000, 64, costs, torch.device("cpu")
    )
    assert budget == 999
    assert activation_memory == 65
    assert costs == _costs(layers_0=(11, 21, 31), head=(1, 6, 1))


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
@patch(f"{_PIPELINE}.log_subgraph_memory")
@patch(f"{_PIPELINE}.plan_subgraph_memory")
@patch(f"{_PIPELINE}.partition_subgraphs")
@patch(f"{_PIPELINE}.find_modules_outside_graph", return_value={})
@patch(f"{_PIPELINE}.trace_graph")
def test_pipeline_auto_partitions_once(
    mock_trace,
    mock_outside,
    mock_partition,
    mock_plan,
    mock_log,
    mock_targets,
    fake_pipeline_inputs,
):
    graph, targets = MagicMock(), {MagicMock()}
    mock_trace.return_value = (graph, targets)
    mock_partition.return_value = [_FakeSubgraph()]
    memory = mock_plan.return_value
    model, dataloader = fake_pipeline_inputs

    SequentialPipeline()(
        model,
        dataloader,
        DatasetArguments(sequential_targets_per_subgraph="auto"),
    )

    # one partition, after the budget is known, with the memory check
    mock_partition.assert_called_once_with(model, graph, targets, is_full=memory)
    assert mock_plan.call_args.args[-1] == 0.9  # gpu_memory_utilization
    mock_log.assert_called_once()


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
def test_pipeline_invalid_targets_per_subgraph(mock_targets, fake_pipeline_inputs):
    model, dataloader = fake_pipeline_inputs

    with pytest.raises(ValueError, match="auto"):
        SequentialPipeline()(
            model, dataloader, DatasetArguments(sequential_targets_per_subgraph="max")
        )


def _run_oneshot_auto(tmp_path, recipe) -> list[str]:
    """Run oneshot with "auto" on a tiny model and return the logged messages"""
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

    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        oneshot(
            model=model,
            # only needed to satisfy oneshot, the dataset is already tokenized
            processor=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"),
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=num_samples,
            sequential_targets_per_subgraph="auto",
        )
    finally:
        logger.remove(sink)

    return messages


@requires_gpu
@pytest.mark.parametrize(
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
def test_oneshot_auto_group_sizes(tmp_path, recipe):
    messages = _run_oneshot_auto(tmp_path, recipe())

    # a tiny model fits in a single subgraph
    assert any("subgraphs of sizes [4]" in message for message in messages)


@requires_gpu
def test_oneshot_auto_falls_back_for_unknown_memory(tmp_path):
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
    messages = _run_oneshot_auto(tmp_path, recipe)

    # SmoothQuant does not estimate its memory, which may grow with the number of
    # targets per subgraph
    assert any(
        "SmoothQuantModifier" in message and "Falling back" in message
        for message in messages
    )
    assert not any("subgraphs of sizes" in message for message in messages)
