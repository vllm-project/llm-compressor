"""Tests for explicit LinearExperts2D -> fused 3D MoE repack (issue #2699)."""

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe
from llmcompressor.utils.dev import skip_weights_initialize


def _tiny_qwen3_vl_moe():
    # 2 * moe_intermediate != hidden_size so HF check_dims Transpose is meaningful
    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
    )
    with skip_weights_initialize():
        model = Qwen3VLMoeForConditionalGeneration(config)
    return model


def test_repack_restores_fused_experts_and_weights():
    model = _tiny_qwen3_vl_moe()
    experts = model.model.language_model.layers[0].mlp.experts
    experts.gate_up_proj.data = torch.arange(
        experts.gate_up_proj.numel(), dtype=torch.float32
    ).reshape_as(experts.gate_up_proj)
    experts.down_proj.data = (
        torch.arange(experts.down_proj.numel(), dtype=torch.float32) + 1000
    ).reshape_as(experts.down_proj)
    ref_gate_up = experts.gate_up_proj.detach().clone()
    ref_down = experts.down_proj.detach().clone()

    linearize_moe(model)
    assert isinstance(model.model.language_model.layers[0].mlp.experts, LinearExperts2D)

    repack_moe(model)
    experts = model.model.language_model.layers[0].mlp.experts
    assert isinstance(experts, FusedExpertsProtocol)
    assert not isinstance(experts, LinearExperts2D)
    assert torch.allclose(experts.gate_up_proj, ref_gate_up)
    assert torch.allclose(experts.down_proj, ref_down)


def test_repack_packs_weight_qparams():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    intermediate = lin.intermediate_size
    hidden = lin[0].gate_proj.in_features

    for i in range(lin.num_experts):
        lin[i].gate_proj.weight_scale = torch.nn.Parameter(
            torch.full((intermediate,), float(i + 1)), requires_grad=False
        )
        lin[i].up_proj.weight_scale = torch.nn.Parameter(
            torch.full((intermediate,), float(i + 10)), requires_grad=False
        )
        lin[i].down_proj.weight_scale = torch.nn.Parameter(
            torch.full((hidden,), float(i + 100)), requires_grad=False
        )

    repack_moe(model)
    experts = model.model.language_model.layers[0].mlp.experts
    assert hasattr(experts, "gate_up_proj_scale")
    assert hasattr(experts, "down_proj_scale")
    assert experts.gate_up_proj_scale.shape == (lin.num_experts, 2 * intermediate)
    assert experts.down_proj_scale.shape == (lin.num_experts, hidden)
    assert torch.allclose(
        experts.gate_up_proj_scale[0, :intermediate],
        torch.full((intermediate,), 1.0),
    )
    assert torch.allclose(
        experts.gate_up_proj_scale[0, intermediate:],
        torch.full((intermediate,), 10.0),
    )


def test_repack_save_pretrained_writes_3d_keys(tmp_path: Path):
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    repack_moe(model)

    out_dir = tmp_path / "repacked"
    model.save_pretrained(out_dir, safe_serialization=True)

    expert_keys = []
    for path in out_dir.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            expert_keys.extend(k for k in handle.keys() if "mlp.experts" in k)

    assert any(k.endswith("experts.gate_up_proj") for k in expert_keys)
    assert any(k.endswith("experts.down_proj") for k in expert_keys)
    assert not any(".experts.0." in k for k in expert_keys)


def test_repack_then_transformers_reload(tmp_path: Path):
    model = _tiny_qwen3_vl_moe()
    experts = model.model.language_model.layers[0].mlp.experts
    experts.gate_up_proj.data = torch.randn_like(experts.gate_up_proj)
    experts.down_proj.data = torch.randn_like(experts.down_proj)
    ref_gate_up = experts.gate_up_proj.detach().clone()
    ref_down = experts.down_proj.detach().clone()

    linearize_moe(model)
    repack_moe(model)
    out_dir = tmp_path / "reload"
    model.save_pretrained(out_dir, safe_serialization=True)

    reloaded = Qwen3VLMoeForConditionalGeneration.from_pretrained(out_dir)
    experts = reloaded.model.language_model.layers[0].mlp.experts
    assert isinstance(experts, FusedExpertsProtocol)
    assert torch.allclose(experts.gate_up_proj, ref_gate_up)
    assert torch.allclose(experts.down_proj, ref_down)
