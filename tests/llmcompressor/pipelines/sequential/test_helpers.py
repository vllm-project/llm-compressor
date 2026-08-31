import math
from functools import wraps

import pytest
import torch
import torch.fx
from transformers import AutoModelForCausalLM, PretrainedConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import (
    get_sequential_ancestors,
    topological_partition,
    trace_subgraphs,
)
from llmcompressor.utils.dev import skip_weights_download, skip_weights_initialize


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


def forward_wrapper(forward):
    @wraps(forward)
    def wrapped(*args, **kwargs):
        return forward(*args, **kwargs)

    return wrapped


class DecoratedDummyModel(torch.nn.Module):
    """Minimal model whose ``forward`` is wrapped in a non-introspectable
    decorator (e.g. ``@can_return_tuple``), the same pattern used by
    ``Qwen3OmniMoeThinkerForConditionalGeneration.forward``."""

    config = PretrainedConfig()
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.layer4 = torch.nn.Linear(10, 10)
        self.layer5 = torch.nn.Linear(10, 10)
        self.layer6 = torch.nn.Linear(10, 10)

    @forward_wrapper
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.layer6(x)


class CombinedConditionModel(torch.nn.Module):
    """``forward`` whose body contains a chained ``if`` test like the
    XOR-input validation in Qwen3-Omni Thinker."""

    config = PretrainedConfig()
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)
        self.input_ids = None

    def forward(self, x, inputs_embeds=None):
        if (x is None) ^ (inputs_embeds is not None):
            raise ValueError("must specify exactly one of x or inputs_embeds")
        if inputs_embeds is None:
            x = self.layer(x)
        return x


class RaiseConditionModel(torch.nn.Module):
    """``forward`` whose body contains an ``if … raise`` block, which the
    autowrap would otherwise execute during FX tracing and kill the trace."""

    config = PretrainedConfig()
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)
        self.input_ids = None

    def forward(self, x):
        if self.input_ids is None:
            raise ValueError("must provide input_ids")
        return self.layer(x)


def test_trace_subgraphs_unwraps_decorated_forward():
    model = DecoratedDummyModel()

    subgraphs = trace_subgraphs(
        model,
        {"x": torch.rand(1, 10)},
        sequential_targets=["Linear"],
        ignore=DatasetArguments().tracing_ignore,
    )

    assert len(subgraphs) == 7


def test_trace_subgraphs_wraps_combined_condition():
    model = CombinedConditionModel()

    subgraphs = trace_subgraphs(
        model,
        {"x": torch.rand(1, 10), "inputs_embeds": None},
        sequential_targets=["Linear"],
        ignore=DatasetArguments().tracing_ignore,
    )

    # The XOR raise check and the conditional layer call are both wrapped
    # into @torch.fx.wrap helpers, so the trace produces at least the
    # preamble subgraph without crashing.
    assert len(subgraphs) >= 1


def test_trace_subgraphs_wraps_raise_condition():
    model = RaiseConditionModel()

    subgraphs = trace_subgraphs(
        model,
        {"x": torch.rand(1, 10)},
        sequential_targets=["Linear"],
        ignore=DatasetArguments().tracing_ignore,
    )

    # The if-raise block is wrapped with body=pass, so FX does not
    # execute the raise and the trace completes with at least the
    # preamble subgraph.
    assert len(subgraphs) >= 1
