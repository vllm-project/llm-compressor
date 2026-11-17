"""
Tests for the default ignore list applied by ModelFreePtqConverter, which
excludes MoE router, lm_head and embedding layers from quantization.
"""

from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.utils.match import match_name

import llmcompressor.entrypoints.model_free as _MODEL_FREE_MODULE
from llmcompressor import model_free_ptq
from llmcompressor.entrypoints.model_free.converter import (
    DEFAULT_IGNORE,
    ModelFreePtqConverter,
)

# Router weight tensors, one per supported naming convention
ROUTER_WEIGHTS = {
    "model.layers.0.mlp.gate.weight": "Qwen, DeepSeek, GLM, Kimi, Cohere, ...",
    "model.layers.1.block_sparse_moe.gate.weight": "Mixtral, Kimi K3, dbrx, ...",
    "model.layers.2.moe.gate.weight": "Step3.5, Grok",
    "backbone.layers.3.mixer.gate.weight": "Nemotron",
    "model.layers.4.mlp.router.weight": "gpt-oss, Trinity",
    "model.layers.5.feed_forward.router.weight": "Llama 4",
    "model.language_model.layers.6.mlp.gate.weight": "multimodal language model",
}

# lm_head and embedding weight tensors, at several nesting depths
LM_HEAD_AND_EMBEDDING_WEIGHTS = {
    "lm_head.weight": "lm_head (Llama, Qwen, Mistral, GLM, gpt-oss, ...)",
    "model.embed_tokens.weight": "embeddings (Llama, Qwen, GLM, gpt-oss, ...)",
    "backbone.embed_tokens.weight": "Nemotron embeddings",
    "model.language_model.embed_tokens.weight": "Llama 4 multimodal embeddings",
    "model.layers.61.embed_tokens.weight": "MTP layer embeddings (DeepSeek, Kimi)",
}

# Linear layers whose names resemble router names but must stay quantizable
LINEAR_WEIGHTS = {
    "model.layers.0.mlp.gate_proj.weight": "GLU gate projection",
    "model.layers.0.mlp.gate_up_proj.weight": "fused gate/up projection",
    "model.layers.1.mlp.experts.3.gate_proj.weight": "per-expert gate projection",
    "model.layers.1.mlp.shared_expert.gate_proj.weight": "shared expert projection",
    "model.layers.2.mlp.shared_expert_gate.weight": "shared expert gate (Qwen MoE)",
    "model.layers.3.self_attn.forget_gate.f_a_proj.weight": "Kimi KDA forget gate",
    "model.layers.3.self_attn.q_proj.weight": "attention projection",
    "model.layers.3.mlp.down_proj.weight": "MLP down projection",
}


def _get_w4a16_config() -> QuantizationConfig:
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="group",
            group_size=16,
            symmetric=True,
            dynamic=False,
        ),
    )
    return QuantizationConfig(config_groups={"group_0": scheme}, ignore=[])


@pytest.mark.parametrize("module_name", sorted(ROUTER_WEIGHTS))
def test_default_ignore_matches_routers(module_name):
    module_name = module_name.removesuffix(".weight")
    assert any(match_name(module_name, pattern) for pattern in DEFAULT_IGNORE)


@pytest.mark.parametrize("module_name", sorted(LM_HEAD_AND_EMBEDDING_WEIGHTS))
def test_default_ignore_matches_lm_head_and_embeddings(module_name):
    module_name = module_name.removesuffix(".weight")
    assert any(match_name(module_name, pattern) for pattern in DEFAULT_IGNORE)


@pytest.mark.parametrize("module_name", sorted(LINEAR_WEIGHTS))
def test_default_ignore_keeps_other_layers_quantizable(module_name):
    module_name = module_name.removesuffix(".weight")
    assert not any(match_name(module_name, pattern) for pattern in DEFAULT_IGNORE)


def test_ignored_layers_pass_through_unquantized():
    """Routers, lm_head and embeddings are skipped, everything else is
    quantized"""
    converter = ModelFreePtqConverter(config=_get_w4a16_config())

    tensors = {
        name: torch.empty((32, 32), dtype=torch.bfloat16, device="meta")
        for name in (
            list(ROUTER_WEIGHTS)
            + list(LM_HEAD_AND_EMBEDDING_WEIGHTS)
            + list(LINEAR_WEIGHTS)
        )
    }
    output = converter.validate(tensors)

    for name in ROUTER_WEIGHTS:
        assert name in output, f"router {name} should not be quantized"
    for name in LM_HEAD_AND_EMBEDDING_WEIGHTS:
        assert name in output, f"{name} should not be quantized"
    for name in LINEAR_WEIGHTS:
        module_name = name.removesuffix(".weight")
        assert name not in output, f"{name} should be quantized"
        assert any(key.startswith(f"{module_name}.weight_") for key in output)


def test_default_ignore_merged_with_user_ignore():
    """User entries are preserved first, defaults appended without duplicates"""
    user_ignore = ["lm_head", r"re:.*mlp\.gate$"]
    config = _get_w4a16_config().model_copy(update={"ignore": user_ignore})
    converter = ModelFreePtqConverter(config=config)

    merged = converter.config.ignore
    assert merged[: len(user_ignore)] == user_ignore
    assert len(merged) == len(set(merged))
    for pattern in DEFAULT_IGNORE:
        assert merged.count(pattern) == 1


def test_default_ignore_does_not_mutate_input_config():
    config = _get_w4a16_config()
    ModelFreePtqConverter(config=config)
    assert config.ignore == []


def test_update_config_records_default_ignore():
    """Defaults are written to the saved quantization config's ignore list"""
    converter = ModelFreePtqConverter(config=_get_w4a16_config())
    output = converter.update_config(None)
    assert set(DEFAULT_IGNORE) <= set(output.ignore)


def test_model_free_ptq_applies_default_ignore(tmp_path):
    """model_free_ptq constructs its converter with the merged ignore list"""
    with (
        patch.object(_MODEL_FREE_MODULE, "get_checkpoint_files", return_value={}),
        patch.object(_MODEL_FREE_MODULE, "validate_safetensors_index"),
        patch.object(_MODEL_FREE_MODULE, "get_weight_map", return_value={}),
        patch.object(_MODEL_FREE_MODULE, "convert_checkpoint") as convert_checkpoint,
    ):
        model_free_ptq("source", tmp_path, scheme="FP8_dynamic", device="cpu")

    converters = convert_checkpoint.call_args.kwargs["converter"]
    mfptq = converters[-1]
    assert isinstance(mfptq, ModelFreePtqConverter)
    assert set(DEFAULT_IGNORE) <= set(mfptq.config.ignore)
