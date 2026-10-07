import json
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
)

import llmcompressor.entrypoints.model_free as _MODEL_FREE_MODULE
from llmcompressor import model_free_ptq
from llmcompressor.entrypoints.model_free import SpinQuantConverter

_COMMON = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)


def _tiny_configs():
    return [
        pytest.param(LlamaConfig(**_COMMON, tie_word_embeddings=False), id="llama"),
        pytest.param(
            Qwen3Config(**_COMMON, head_dim=32, tie_word_embeddings=True),
            id="qwen3-tied",
        ),
        pytest.param(
            Qwen2Config(**_COMMON, tie_word_embeddings=False), id="qwen2-bias"
        ),
        pytest.param(
            Qwen3MoeConfig(
                **_COMMON,
                head_dim=16,
                moe_intermediate_size=32,
                num_experts=4,
                num_experts_per_tok=2,
                tie_word_embeddings=False,
            ),
            id="qwen3-moe",
        ),
    ]


def _save_tiny_model(config, path) -> torch.nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("bias"):
                param.normal_(std=0.1)
    model.eval()
    model.save_pretrained(path)
    return model


@pytest.mark.parametrize("config", _tiny_configs())
@pytest.mark.parametrize("transform_block_size", (None, 32))
def test_offline_rotations_preserve_outputs(config, transform_block_size, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    model = _save_tiny_model(config, source)

    converter = SpinQuantConverter.from_pretrained(
        source, transform_block_size=transform_block_size
    )
    convert_checkpoint(source, target, converter, device="cpu")

    rotated = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32)
    rotated.eval()
    assert not rotated.config.tie_word_embeddings

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = rotated(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    # rotation must actually change the weights, and norms are fused away
    rotated_params = dict(rotated.named_parameters())
    q_proj = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.allclose(rotated_params[q_proj], model.get_parameter(q_proj))
    for name, param in rotated_params.items():
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            assert torch.equal(param, torch.ones_like(param))

    with open(target / "config.json") as file:
        assert json.load(file)["tie_word_embeddings"] is False


def test_fused_3d_experts_match_2d_experts():
    hidden, intermediate, num_experts = 32, 16, 3
    norms = {"model.layers.0.post_attention_layernorm": torch.rand(hidden) + 0.5}
    converter = SpinQuantConverter(hidden_size=hidden, head_dim=16, norms=norms)

    gate_up = torch.randn(num_experts, 2 * intermediate, hidden)
    down = torch.randn(num_experts, hidden, intermediate)
    prefix = "model.layers.0.mlp.experts"
    fused = converter.process(
        {f"{prefix}.gate_up_proj": gate_up, f"{prefix}.down_proj": down}
    )
    for expert in range(num_experts):
        split = converter.process(
            {
                f"{prefix}.{expert}.gate_proj.weight": gate_up[expert, :intermediate],
                f"{prefix}.{expert}.up_proj.weight": gate_up[expert, intermediate:],
                f"{prefix}.{expert}.down_proj.weight": down[expert],
            }
        )
        torch.testing.assert_close(
            fused[f"{prefix}.gate_up_proj"][expert],
            torch.cat(
                [
                    split[f"{prefix}.{expert}.gate_proj.weight"],
                    split[f"{prefix}.{expert}.up_proj.weight"],
                ]
            ),
        )
        torch.testing.assert_close(
            fused[f"{prefix}.down_proj"][expert],
            split[f"{prefix}.{expert}.down_proj.weight"],
        )


def test_rejects_quantized_tensors():
    converter = SpinQuantConverter(hidden_size=32, head_dim=16, norms={})
    with pytest.raises(ValueError, match="dequantize"):
        converter.process(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                    32, 32, dtype=torch.float8_e4m3fn
                ),
                "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(1, 1),
            }
        )


def test_rejects_unmapped_residual_layers():
    converter = SpinQuantConverter(hidden_size=32, head_dim=16, norms={})
    with pytest.raises(ValueError, match="not covered"):
        converter.process({"model.layers.0.mlp.other.weight": torch.randn(8, 32)})


def test_rejects_norm_bias():
    norm = "model.layers.0.input_layernorm"
    converter = SpinQuantConverter(
        hidden_size=32, head_dim=16, norms={norm: torch.ones(32)}
    )
    with pytest.raises(ValueError, match="norms without bias"):
        converter.process({f"{norm}.bias": torch.zeros(32)})


def test_ignore_skips_tensors():
    converter = SpinQuantConverter(
        hidden_size=32, head_dim=16, norms={}, ignore=[r"^visual\."]
    )
    tensor = torch.randn(8, 32)
    assert converter.process({"visual.proj.weight": tensor})["visual.proj.weight"] is (
        tensor
    )


def test_model_free_ptq_chains_converter_list(tmp_path):
    first, second = object(), object()
    with (
        patch.object(_MODEL_FREE_MODULE, "get_checkpoint_files", return_value={}),
        patch.object(_MODEL_FREE_MODULE, "validate_safetensors_index"),
        patch.object(_MODEL_FREE_MODULE, "get_weight_map", return_value={}),
        patch.object(_MODEL_FREE_MODULE, "convert_checkpoint") as convert_checkpoint,
    ):
        model_free_ptq(
            "source",
            tmp_path,
            scheme="FP8_dynamic",
            converter=[first, second],
            device="cpu",
        )

    converters = convert_checkpoint.call_args.kwargs["converter"]
    assert converters[:2] == [first, second]
    assert isinstance(converters[2], _MODEL_FREE_MODULE.ModelFreePtqConverter)
