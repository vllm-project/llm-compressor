"""Tests for explicit LinearExperts2D -> fused 3D MoE repack (issues #2699, #3183)."""

from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils import replace_direct_state_dict
from safetensors import safe_open
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    WeightRenaming,
    revert_weight_conversion,
)

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.modeling.moe.linear_experts import (
    CompressedFusedLinear,
    LinearExperts2D,
)
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


def _mark_linear_compressed(linear: torch.nn.Linear, state: dict[str, torch.Tensor]):
    replace_direct_state_dict(linear, state)
    linear.quantization_status = QuantizationStatus.COMPRESSED


def _nvfp4_like_state(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: float,
    input_global_scale: float,
) -> dict[str, torch.Tensor]:
    return {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_global_scale": torch.tensor([global_scale]),
        "input_global_scale": torch.tensor([input_global_scale]),
    }


def test_repack_packs_compressed_nested_modules():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    num_experts = lin.num_experts
    hidden = lin[0].gate_proj.in_features
    intermediate = lin.intermediate_size
    packed_in = hidden // 2
    scale_in = max(hidden // 16, 1)

    for i in range(num_experts):
        gate_state = _nvfp4_like_state(
            torch.full((intermediate, packed_in), i + 1, dtype=torch.uint8),
            torch.full((intermediate, scale_in), float(i + 1)),
            float(i + 1),
            float(i + 2),
        )
        gate_state["bias"] = torch.full((intermediate,), float(i))
        _mark_linear_compressed(lin[i].gate_proj, gate_state)
        up_state = _nvfp4_like_state(
            torch.full((intermediate, packed_in), i + 10, dtype=torch.uint8),
            torch.full((intermediate, scale_in), float(i + 10)),
            float(i + 1),
            float(i + 2),
        )
        up_state["bias"] = torch.full((intermediate,), float(i + 10))
        _mark_linear_compressed(lin[i].up_proj, up_state)
        down_state = _nvfp4_like_state(
            torch.full((hidden, max(intermediate // 2, 1)), i + 100, dtype=torch.uint8),
            torch.full((hidden, max(intermediate // 16, 1)), float(i + 100)),
            float(i + 6),
            float(i + 7),
        )
        down_state["bias"] = torch.full((hidden,), float(i + 100))
        _mark_linear_compressed(lin[i].down_proj, down_state)

    repack_moe(model)
    experts = model.model.language_model.layers[0].mlp.experts
    assert isinstance(experts.gate_up_proj, CompressedFusedLinear)
    assert isinstance(experts.down_proj, CompressedFusedLinear)

    packed_gate_up = experts.gate_up_proj.weight_packed
    assert packed_gate_up.shape == (num_experts, 2 * intermediate, packed_in)
    assert torch.equal(
        packed_gate_up[0, :intermediate],
        torch.full((intermediate, packed_in), 1, dtype=torch.uint8),
    )
    assert torch.equal(
        packed_gate_up[0, intermediate:],
        torch.full((intermediate, packed_in), 10, dtype=torch.uint8),
    )
    assert experts.gate_up_proj.weight_global_scale.shape == (num_experts, 2)
    assert torch.equal(
        experts.gate_up_proj.weight_global_scale[0], torch.tensor([1.0, 1.0])
    )
    assert torch.equal(
        experts.gate_up_proj.input_global_scale[0], torch.tensor([2.0, 2.0])
    )

    keys = set(experts.state_dict())
    assert "gate_up_proj.weight_packed" in keys
    assert "down_proj.weight_packed" in keys
    assert "gate_up_proj.weight_scale" in keys
    assert "down_proj.input_global_scale" in keys
    assert "gate_up_proj.bias" not in keys
    assert "down_proj.bias" not in keys
    assert not any(key.startswith("0.") for key in keys)


def test_repack_rejects_quantized_but_uncompressed_experts():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    for i in range(lin.num_experts):
        lin[i].gate_proj.quantization_status = QuantizationStatus.FROZEN
        lin[i].up_proj.quantization_status = QuantizationStatus.FROZEN
        lin[i].down_proj.quantization_status = QuantizationStatus.FROZEN

    with pytest.raises(RuntimeError, match="before they are compressed"):
        repack_moe(model)


def test_repack_rejects_mismatched_compressed_keys():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    packed = torch.ones((lin.intermediate_size, 32), dtype=torch.uint8)
    scale = torch.ones((lin.intermediate_size, 4))
    down_packed = torch.ones((64, 8), dtype=torch.uint8)
    down_scale = torch.ones((64, 1))

    for i in range(lin.num_experts):
        gate_state = {"weight_packed": packed.clone(), "weight_scale": scale.clone()}
        if i == 1:
            gate_state.pop("weight_scale")
        _mark_linear_compressed(lin[i].gate_proj, gate_state)
        _mark_linear_compressed(
            lin[i].up_proj,
            {"weight_packed": packed.clone(), "weight_scale": scale.clone()},
        )
        _mark_linear_compressed(
            lin[i].down_proj,
            {"weight_packed": down_packed.clone(), "weight_scale": down_scale.clone()},
        )

    with pytest.raises(RuntimeError, match="mismatched"):
        repack_moe(model)


def test_repack_packs_ungated_compressed_nested_modules():
    class UngatedFusedExperts(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            num_experts = config.num_experts
            hidden = config.hidden_size
            intermediate = config.moe_intermediate_size
            self.up_proj = torch.nn.Parameter(
                torch.empty(num_experts, intermediate, hidden)
            )
            self.down_proj = torch.nn.Parameter(
                torch.empty(num_experts, hidden, intermediate)
            )

    class UngatedLinearExperts(LinearExperts2D):
        is_concatenated = True
        is_transposed = False
        has_bias = False
        has_gate = False
        _apply_gate = staticmethod(lambda x: x)

    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 128,
            "hidden_act": "silu",
        }
    ).text_config
    config.dtype = torch.float32
    lin = UngatedLinearExperts(config)
    lin._source_experts_cls = UngatedFusedExperts
    lin._source_config = config
    num_experts = lin.num_experts
    hidden = lin[0].up_proj.in_features
    intermediate = lin.intermediate_size
    packed_in = hidden // 2

    for i in range(num_experts):
        _mark_linear_compressed(
            lin[i].up_proj,
            _nvfp4_like_state(
                torch.full((intermediate, packed_in), i + 1, dtype=torch.uint8),
                torch.full((intermediate, max(hidden // 16, 1)), float(i + 1)),
                float(i + 1),
                float(i + 2),
            ),
        )
        _mark_linear_compressed(
            lin[i].down_proj,
            _nvfp4_like_state(
                torch.full(
                    (hidden, max(intermediate // 2, 1)), i + 100, dtype=torch.uint8
                ),
                torch.full((hidden, max(intermediate // 16, 1)), float(i + 100)),
                float(i + 6),
                float(i + 7),
            ),
        )

    fused = lin.to_experts_module()
    assert isinstance(fused.up_proj, CompressedFusedLinear)
    assert isinstance(fused.down_proj, CompressedFusedLinear)
    assert fused.up_proj.weight_packed.shape == (
        num_experts,
        intermediate,
        packed_in,
    )
    assert torch.equal(
        fused.up_proj.weight_packed[0],
        torch.full((intermediate, packed_in), 1, dtype=torch.uint8),
    )
    assert fused.up_proj.weight_global_scale.shape == (num_experts, 1)
    assert torch.equal(fused.up_proj.weight_global_scale[0], torch.tensor([1.0]))
    keys = set(fused.state_dict())
    assert "up_proj.weight_packed" in keys
    assert "down_proj.weight_packed" in keys
    assert "up_proj.weight_scale" in keys
    assert "down_proj.input_global_scale" in keys
    assert not any(key.startswith("0.") for key in keys)


def test_repack_rejects_mixed_compressed_and_uncompressed():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    packed = torch.ones((lin.intermediate_size, 32), dtype=torch.uint8)
    scale = torch.ones((lin.intermediate_size, 4))
    _mark_linear_compressed(
        lin[0].gate_proj,
        {"weight_packed": packed.clone(), "weight_scale": scale.clone()},
    )
    with pytest.raises(RuntimeError, match="mix of compressed and uncompressed"):
        repack_moe(model)


def test_repack_rejects_compressed_status_with_dense_weight():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    for i in range(lin.num_experts):
        for proj in (lin[i].gate_proj, lin[i].up_proj, lin[i].down_proj):
            proj.quantization_status = QuantizationStatus.COMPRESSED
    with pytest.raises(RuntimeError, match="still have a dense"):
        repack_moe(model)


def test_repack_save_conversion_emits_inkling_nested_keys():
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    lin = model.model.language_model.layers[0].mlp.experts
    hidden = lin[0].gate_proj.in_features
    intermediate = lin.intermediate_size
    packed_in = hidden // 2

    for i in range(lin.num_experts):
        _mark_linear_compressed(
            lin[i].gate_proj,
            _nvfp4_like_state(
                torch.full((intermediate, packed_in), i + 1, dtype=torch.uint8),
                torch.ones((intermediate, max(hidden // 16, 1))),
                1.0,
                2.0,
            ),
        )
        _mark_linear_compressed(
            lin[i].up_proj,
            _nvfp4_like_state(
                torch.full((intermediate, packed_in), i + 10, dtype=torch.uint8),
                torch.ones((intermediate, max(hidden // 16, 1))) * 2,
                3.0,
                4.0,
            ),
        )
        _mark_linear_compressed(
            lin[i].down_proj,
            _nvfp4_like_state(
                torch.full(
                    (hidden, max(intermediate // 2, 1)), i + 100, dtype=torch.uint8
                ),
                torch.ones((hidden, max(intermediate // 16, 1))) * 5,
                6.0,
                7.0,
            ),
        )

    repack_moe(model)
    experts = model.model.language_model.layers[0].mlp.experts
    state = {
        f"model.llm.layers.0.mlp.experts.{name}": tensor
        for name, tensor in experts.state_dict().items()
    }
    model._weight_conversions = [
        WeightRenaming(
            source_patterns=r"mlp.experts.w13_weight",
            target_patterns=r"mlp.experts.gate_up_proj",
        ),
        WeightRenaming(
            source_patterns=r"mlp.experts.w2_weight",
            target_patterns=r"mlp.experts.down_proj",
        ),
    ]
    converted = revert_weight_conversion(model, state)
    converted_keys = set(converted)
    expected = {
        "model.llm.layers.0.mlp.experts.w13_weight.weight_packed",
        "model.llm.layers.0.mlp.experts.w13_weight.weight_scale",
        "model.llm.layers.0.mlp.experts.w13_weight.weight_global_scale",
        "model.llm.layers.0.mlp.experts.w13_weight.input_global_scale",
        "model.llm.layers.0.mlp.experts.w2_weight.weight_packed",
        "model.llm.layers.0.mlp.experts.w2_weight.weight_scale",
        "model.llm.layers.0.mlp.experts.w2_weight.weight_global_scale",
        "model.llm.layers.0.mlp.experts.w2_weight.input_global_scale",
    }
    assert expected <= converted_keys
    assert not any(".experts.0." in key for key in converted_keys)
    mapping = get_checkpoint_conversion_mapping("inkling_mm_model")
    if mapping is not None:
        assert any("w13_weight" in str(item) for item in mapping)


def test_llama4_from_experts_accepts_text_config():
    from transformers.models.llama4.configuration_llama4 import (
        Llama4Config,
        Llama4TextConfig,
    )
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

    from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts

    text_config = Llama4TextConfig(
        hidden_size=64,
        intermediate_size=32,
        num_local_experts=2,
        num_experts_per_tok=1,
    )
    experts = Llama4TextExperts(text_config)
    from_text = Llama4LinearExperts.from_experts_module(experts, text_config)
    assert from_text._source_config is text_config

    parent = Llama4Config(text_config=text_config)
    from_parent = Llama4LinearExperts.from_experts_module(experts, parent)
    assert from_parent._source_config is parent.text_config
