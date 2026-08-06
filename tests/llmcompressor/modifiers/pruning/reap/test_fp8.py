import json
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from compressed_tensors.offload.module import offload_module
from safetensors.torch import load_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.integrations.finegrained_fp8 import FP8Experts
from transformers.utils.quantization_config import FineGrainedFP8Config

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling.moe.fp8_experts import (
    FP8PrunableExperts,
    make_fp8_experts_reap_prunable,
)
from llmcompressor.modeling.moe.helpers import ReapPrunableExpertsProtocol
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.modifiers.pruning.reap.utils import (
    REAPSaliencyTracker,
    get_moe_attrs,
    prune_moe_layer,
    update_model_config,
)


def _config(num_experts: int = 4) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        hidden_size=4,
        moe_intermediate_size=2,
        n_routed_experts=num_experts,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
    )


def _tiny_glm_config(num_experts: int = 4) -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=num_experts,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        indexer_types=["full"],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        kv_lora_rank=4,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=32,
    )
    config.quantization_config = FineGrainedFP8Config(
        weight_block_size=(2, 2),
        activation_scheme="dynamic",
        modules_to_not_convert=["self_attn", "shared_experts", "lm_head"],
    )
    return config


def _cpu_linear(
    self,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # CPU test substitute for the CUDA FP8 kernel. The expert forward and
    # packed FP8 storage remain the same as the Transformers implementation.
    del weight_scale_inv, activation_scale
    return F.linear(input, weight.float())


def _fp8_experts(
    config: GlmMoeDsaConfig,
    activation_scheme: str = "dynamic",
    has_gate: bool = True,
) -> FP8Experts:
    experts = FP8Experts(
        config,
        block_size=(2, 2),
        activation_scheme=activation_scheme,
        has_gate=has_gate,
    )
    with torch.no_grad():
        torch.manual_seed(0)
        input_projection = experts.gate_up_proj if has_gate else experts.up_proj
        input_projection.copy_(
            torch.randn(input_projection.shape).to(input_projection.dtype)
        )
        experts.down_proj.copy_(
            torch.randn(experts.down_proj.shape).to(experts.down_proj.dtype)
        )
        input_scale = (
            experts.gate_up_proj_scale_inv if has_gate else experts.up_proj_scale_inv
        )
        input_scale.fill_(1.0)
        experts.down_proj_scale_inv.fill_(1.0)
    experts.linear = types.MethodType(_cpu_linear, experts)
    return experts


class _FP8Router(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(
            torch.arange(
                config.n_routed_experts * config.hidden_size,
                dtype=torch.float32,
            ).reshape(config.n_routed_experts, config.hidden_size)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.arange(config.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class _FP8MoEBlock(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, activation_scheme="dynamic"):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.gate = _FP8Router(config)
        self.experts = _fp8_experts(config, activation_scheme)
        self.register_buffer(
            "e_score_correction_bias",
            torch.arange(config.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits = self.gate(hidden_states)
        router_probabilities = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            router_probabilities, k=self.top_k, dim=-1
        )
        return self.experts(hidden_states, top_k_indices, top_k_weights).reshape(
            original_shape
        )


class _FP8MoEModel(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, activation_scheme="dynamic"):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([_FP8MoEBlock(config, activation_scheme)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@pytest.mark.unit
def test_fp8_experts_collect_unweighted_reap_norms():
    model = _FP8MoEModel(_config())
    get_moe_attrs(model, ignore=[])
    experts = model.layers[0].experts

    assert isinstance(experts, FP8PrunableExperts)
    assert isinstance(experts, ReapPrunableExpertsProtocol)

    hidden_states = torch.tensor(
        [[1.0, 0.0, -1.0, 0.5], [0.5, 1.0, 0.0, -0.5], [1.0, 1.0, 1.0, 1.0]]
    )
    top_k_indices = torch.tensor([[0, 1], [2, 0], [3, 1]])
    top_k_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]])

    experts.start_reap_norm_collection()
    experts(hidden_states, top_k_indices, top_k_weights)
    norms = experts.take_reap_norms()

    assert set(norms) == {0, 1, 2, 3}
    assert {index: len(value) for index, value in norms.items()} == {
        0: 2,
        1: 2,
        2: 1,
        3: 1,
    }

    tracker = REAPSaliencyTracker(num_experts=4)
    tracker.update(top_k_indices, top_k_weights, norms)
    flat_indices = top_k_indices.T.reshape(-1)
    flat_weights = top_k_weights.T.reshape(-1)
    for expert_index, expert_norms in norms.items():
        expected = (flat_weights[flat_indices == expert_index] * expert_norms).mean()
        assert tracker.mean_saliency[expert_index].item() == pytest.approx(
            expected.item()
        )


@pytest.mark.unit
@pytest.mark.parametrize("activation_scheme", ["dynamic", "static"])
@pytest.mark.parametrize("has_gate", [True, False])
def test_reap_norm_collection_preserves_fp8_forward(activation_scheme, has_gate):
    experts = _fp8_experts(
        _config(), activation_scheme=activation_scheme, has_gate=has_gate
    )
    hidden_states = torch.tensor(
        [[1.0, 0.0, -1.0, 0.5], [0.5, 1.0, 0.0, -0.5], [1.0, 1.0, 1.0, 1.0]]
    )
    top_k_indices = torch.tensor([[0, 1], [2, 0], [3, 1]])
    top_k_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]])

    expected = experts(hidden_states, top_k_indices, top_k_weights)
    make_fp8_experts_reap_prunable(experts)
    experts.start_reap_norm_collection()
    actual = experts(hidden_states, top_k_indices, top_k_weights)

    torch.testing.assert_close(actual, expected)
    assert set(experts.take_reap_norms()) == {0, 1, 2, 3}


@pytest.mark.unit
def test_fp8_adapter_refreshes_compressed_tensors_offload_forward():
    experts = _fp8_experts(_config())
    offload_module(experts, onload_device="cpu", offload_device="cpu")
    original_weight = experts.gate_up_proj.detach().clone()
    original_scale = experts.gate_up_proj_scale_inv.detach().clone()

    make_fp8_experts_reap_prunable(experts)

    assert isinstance(experts, FP8PrunableExperts)
    assert experts._original_forward_func is FP8PrunableExperts.forward

    hidden_states = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    top_k_indices = torch.tensor([[0, 1]])
    top_k_weights = torch.tensor([[0.75, 0.25]])
    experts.start_reap_norm_collection()
    experts(hidden_states, top_k_indices, top_k_weights)

    assert set(experts.take_reap_norms()) == {0, 1}

    retained = [3, 1]
    experts.prune_experts_(retained)
    torch.testing.assert_close(experts.gate_up_proj, original_weight[retained])
    torch.testing.assert_close(experts.gate_up_proj_scale_inv, original_scale[retained])


@pytest.mark.unit
def test_fp8_adapter_refreshes_accelerate_offload_forward():
    experts = _fp8_experts(_config())
    add_hook_to_module(experts, AlignDevicesHook(execution_device="cpu"))

    make_fp8_experts_reap_prunable(experts)

    assert isinstance(experts, FP8PrunableExperts)
    assert experts._old_forward.__func__ is FP8PrunableExperts.forward

    hidden_states = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    top_k_indices = torch.tensor([[0, 1]])
    top_k_weights = torch.tensor([[0.75, 0.25]])
    experts.start_reap_norm_collection()
    experts(hidden_states, top_k_indices, top_k_weights)

    assert set(experts.take_reap_norms()) == {0, 1}


@pytest.mark.unit
def test_prune_fp8_experts_slices_weights_scales_router_and_config():
    model = _FP8MoEModel(_config(), activation_scheme="static")
    attrs = get_moe_attrs(model, ignore=[])
    layer = model.layers[0]
    experts = layer.experts
    retained = [3, 1]

    tensor_names = (
        "gate_up_proj",
        "gate_up_proj_scale_inv",
        "down_proj",
        "down_proj_scale_inv",
        "gate_up_proj_activation_scale",
        "down_proj_activation_scale",
    )
    originals = {name: getattr(experts, name).detach().clone() for name in tensor_names}
    original_router_weight = layer.gate.weight.detach().clone()

    prune_moe_layer(model, attrs.moe_layer_names[0], retained, attrs)
    update_model_config(model, attrs, len(retained))

    assert experts.num_experts == len(retained)
    for name, original in originals.items():
        pruned = getattr(experts, name)
        assert pruned.dtype == original.dtype
        torch.testing.assert_close(pruned, original[retained])

    torch.testing.assert_close(layer.gate.weight, original_router_weight[retained])
    torch.testing.assert_close(
        layer.gate.e_score_correction_bias,
        torch.tensor(retained, dtype=torch.float32),
    )
    torch.testing.assert_close(
        layer.e_score_correction_bias,
        torch.tensor(retained, dtype=torch.float32),
    )
    assert layer.n_routed_experts == len(retained)
    assert layer.gate.n_routed_experts == len(retained)
    assert model.config.n_routed_experts == len(retained)
    assert model.config.num_local_experts == len(retained)

    state_keys = set(experts.state_dict())
    assert state_keys == {
        "gate_up_proj",
        "gate_up_proj_scale_inv",
        "down_proj",
        "down_proj_scale_inv",
        "gate_up_proj_activation_scale",
        "down_proj_activation_scale",
    }
    assert attrs.num_experts_config_key == "n_routed_experts"


@pytest.mark.unit
def test_native_fp8_adapter_rejects_non_fp8_weight_storage():
    model = _FP8MoEModel(_config())
    experts = model.layers[0].experts
    experts.gate_up_proj = nn.Parameter(experts.gate_up_proj.detach().float())

    with pytest.raises(TypeError, match="only supports e4m3 FP8"):
        get_moe_attrs(model, ignore=[])


@pytest.mark.unit
def test_fp8_reap_modifier_full_lifecycle():
    model = _FP8MoEModel(_config())
    modifier = REAPPruningModifier(sparsity=0.5)
    state = State(
        model=model,
        teacher_model=None,
        optimizer=None,
        optim_wrapped=False,
        loss=None,
        batch_data=None,
    )

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        model(torch.randn(2, 3, model.config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    layer = model.layers[0]
    assert layer.experts.num_experts == 2
    assert layer.experts.gate_up_proj.shape[0] == 2
    assert layer.gate.weight.shape[0] == 2
    assert layer.e_score_correction_bias.shape[0] == 2
    assert layer.gate.e_score_correction_bias.shape[0] == 2
    assert model.config.n_routed_experts == 2
    with torch.no_grad():
        output = model(torch.randn(2, 3, model.config.hidden_size))
    assert output.shape == (2, 3, model.config.hidden_size)
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_pruned_fp8_glm_saves_and_reloads_with_transformers(tmp_path):
    config = _tiny_glm_config()
    model = GlmMoeDsaForCausalLM(config)
    model.model.layers[0].mlp.experts = _fp8_experts(config)

    attrs = get_moe_attrs(model, ignore=[])
    prune_moe_layer(model, attrs.moe_layer_names[0], [3, 1], attrs)
    update_model_config(model, attrs, new_num_experts=2)
    model.save_pretrained(tmp_path)

    with open(tmp_path / "config.json") as config_file:
        saved_config = json.load(config_file)
    assert saved_config["n_routed_experts"] == 2
    assert saved_config["quantization_config"]["quant_method"] == "fp8"

    saved_state = load_file(tmp_path / "model.safetensors")
    expert_weights = {
        name: tensor
        for name, tensor in saved_state.items()
        if ".mlp.experts." in name and name.endswith(".weight")
    }
    expert_scales = {
        name: tensor
        for name, tensor in saved_state.items()
        if ".mlp.experts." in name and name.endswith(".weight_scale_inv")
    }
    assert len(expert_weights) == 6
    assert len(expert_scales) == 6
    assert all(
        tensor.dtype == torch.float8_e4m3fn for tensor in expert_weights.values()
    )

    reloaded, loading_info = GlmMoeDsaForCausalLM.from_pretrained(
        tmp_path, output_loading_info=True
    )
    assert reloaded.config.n_routed_experts == 2
    assert reloaded.model.layers[0].mlp.experts.gate_up_proj.shape[0] == 2
    assert not [key for key in loading_info["missing_keys"] if ".mlp.experts." in key]
    assert not [
        key for key in loading_info["unexpected_keys"] if ".mlp.experts." in key
    ]
