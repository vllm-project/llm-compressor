from types import SimpleNamespace

import pytest
import torch
from transformers import initialization as init
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

import llmcompressor.modeling.moe.linearize as linearize_mod
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    get_moe_linear_status,
    linearize_moe_layer,
    linearize_moe_subgraph,
    load_quantizable_moe,
)


class _DummyOnloadWrapper:
    def __init__(self):
        self.replaced_with = None

    def replace_with(self, new_module):
        self.replaced_with = new_module


class _TwoExpertBlocks(torch.nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config

        self.block1 = torch.nn.Module()
        self.block1.mlp = torch.nn.Module()
        self.block1.mlp.experts = Qwen3MoeExperts(config)

        self.block2 = torch.nn.Module()
        self.block2.mlp = torch.nn.Module()
        self.block2.mlp.experts = Qwen3MoeExperts(config)

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)


def _make_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )


def _init_experts(experts: Qwen3MoeExperts, config: Qwen3MoeConfig) -> None:
    init.normal_(experts.gate_up_proj, mean=0.0, std=config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)


@torch.no_grad()
def test_linearize_moe_layer_replaces_module_and_updates_lookup():
    config = _make_config()
    model = _TwoExpertBlocks(config)
    experts = model.block1.mlp.experts
    _init_experts(experts, config)

    get_moe_linear_status(model)
    wrapper = _DummyOnloadWrapper()
    experts._onload_wrapper = wrapper

    linearize_moe_layer(model, "block1.mlp.experts", experts)

    new_module = model.block1.mlp.experts
    assert new_module is not experts
    assert isinstance(new_module, LinearExperts2D)
    assert wrapper.replaced_with is new_module
    assert experts not in model._moe_lookup
    assert model._moe_lookup[new_module] == "block1.mlp.experts"


@torch.no_grad()
def test_linearize_moe_subgraph_only_targets_selected_modules(monkeypatch):
    config = _make_config()
    model = _TwoExpertBlocks(config)
    _init_experts(model.block1.mlp.experts, config)
    _init_experts(model.block2.mlp.experts, config)

    calls = []

    def fake_linearize_moe_layer(model_arg, name, module):
        calls.append((name, module))

    monkeypatch.setattr(linearize_mod, "linearize_moe_layer", fake_linearize_moe_layer)

    linearize_moe_subgraph(
        model,
        [
            model.block1.mlp.experts,
            model.dense,
        ],
    )

    assert calls == [("block1.mlp.experts", model.block1.mlp.experts)]


@torch.no_grad()
def test_linearize_moe_subgraph_promotes_selected_expert_children(monkeypatch):
    config = _make_config()
    model = _TwoExpertBlocks(config)
    _init_experts(model.block1.mlp.experts, config)

    calls = []

    def fake_linearize_moe_layer(model_arg, name, module):
        calls.append((name, module))

    monkeypatch.setattr(linearize_mod, "linearize_moe_layer", fake_linearize_moe_layer)

    expert_child = next(iter(model.block1.mlp.experts.children()))
    linearize_moe_subgraph(model, [expert_child])

    assert calls == [("block1.mlp.experts", model.block1.mlp.experts)]


@torch.no_grad()
def test_linearize_moe_layer_rejects_non_moe_module():
    config = _make_config()
    model = _TwoExpertBlocks(config)
    non_moe = model.dense

    with pytest.raises(ValueError):
        linearize_moe_layer(model, "dense", non_moe)


@torch.no_grad()
def test_load_quantizable_moe_fallback_leaves_model_unlinearized(monkeypatch):
    config = _make_config()
    model = _TwoExpertBlocks(config)

    class _DummyLoader:
        from_pretrained_calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_calls += 1
            return model

    monkeypatch.setattr(
        linearize_mod.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="dummy_moe"),
    )
    monkeypatch.setattr(linearize_mod, "has_linearize_load_mappings", lambda *_: False)

    with load_quantizable_moe(_DummyLoader):
        loaded = _DummyLoader.from_pretrained("ignored")

    assert loaded is model
    assert _DummyLoader.from_pretrained_calls == 1
    assert len([m for m in loaded.modules() if isinstance(m, LinearExperts2D)]) == 0
    assert len([m for m in loaded.modules() if isinstance(m, Qwen3MoeExperts)]) == 2
