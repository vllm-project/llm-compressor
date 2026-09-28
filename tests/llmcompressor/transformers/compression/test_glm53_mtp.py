import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors.torch import load_file, save_file
from transformers import FineGrainedFP8Config, GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from llmcompressor import oneshot, reset_session
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.mtp import load_mtp_model
from llmcompressor.utils import load_context

PREFIX = "model.layers.2."
IGNORE = [
    "lm_head",
    r"re:.*\.eh_proj$",
    r"re:.*\.mlp\.gate$",
    r"re:.*\.indexer\.(?:wk|weights_proj)$",
]


def _source_model(tmp_path, native_fp8):
    torch.manual_seed(42)
    config = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        num_nextn_predict_layers=1,
        indexer_types=["full", "full"],
    )
    model = GlmMoeDsaForCausalLM(config).bfloat16().eval()
    linearize_moe(model)
    model.bfloat16()
    source = tmp_path / "source"
    model.save_pretrained(source)
    weights = load_file(source / "model.safetensors")
    # GLM-5.3 stores an appended MoE decoder plus these MTP projections.
    for name, value in list(weights.items()):
        if name.startswith("model.layers.1."):
            weights[name.replace("model.layers.1.", PREFIX)] = value.clone()
    for name in ("enorm", "hnorm", "shared_head.norm"):
        weights[f"{PREFIX}{name}.weight"] = torch.ones(128, dtype=torch.bfloat16)
    weights[f"{PREFIX}eh_proj.weight"] = torch.randn(128, 256).bfloat16()
    expected = {
        name: value.clone()
        for name, value in weights.items()
        if name.startswith(PREFIX)
    }
    kwargs = {}
    if native_fp8:
        for name, value in list(weights.items()):
            if (
                value.ndim != 2
                or not name.endswith(".weight")
                or any(
                    part in name
                    for part in (
                        "embed_tokens",
                        "lm_head",
                        "eh_proj",
                        "mlp.gate",
                        "indexer.weights_proj",
                    )
                )
            ):
                continue
            weights[name] = value.to(torch.float8_e4m3fn)
            shape = tuple((dim + 127) // 128 for dim in value.shape)
            scale = (torch.arange(shape[0] * shape[1]).reshape(shape) + 1) * 0.01317
            weights[name.replace(".weight", ".weight_scale_inv")] = scale
            if name in expected:
                expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
                expected[name] = (
                    weights[name].float() * expanded[: value.shape[0], : value.shape[1]]
                ).bfloat16()
        config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                name.removesuffix(".weight")
                for name, value in weights.items()
                if name.endswith(".weight")
                and value.ndim == 2
                and value.dtype != torch.float8_e4m3fn
            ],
        }
        config.save_pretrained(source)
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    save_file(weights, source / "model.safetensors")
    with load_context(GlmMoeDsaForCausalLM):
        model = GlmMoeDsaForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, **kwargs
        )
    # The real checkpoint uses layer 78; the scaled fixture uses layer 2.
    model._keys_to_ignore_on_load_unexpected = [r"model\.layers\.2.*"]
    return model, expected


@pytest.mark.parametrize("native_fp8", [False, True])
@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "MXFP4", "NVFP4A16"])
def test_glm53_mtp_roundtrip(tmp_path, native_fp8, scheme):
    model, expected = _source_model(tmp_path, native_fp8)
    load_mtp_model(model)
    assert len(model.mtp.layers) == 1
    assert model.mtp.config.mlp_layer_types == ["sparse"]
    for name, value in model.mtp.layers[0].named_parameters():
        source = PREFIX + name.removeprefix("mtp_block.").replace(
            "post_norm.", "shared_head.norm."
        )
        torch.testing.assert_close(value, expected[source], rtol=0, atol=0)

    recipe = QuantizationModifier(
        scheme=scheme, targets=["Linear", r"re:^mtp\.layers\."], ignore=IGNORE
    )
    reset_session()
    oneshot(model=model, recipe=recipe)
    destination = tmp_path / "saved"
    model.save_pretrained(destination)
    output = load_file(destination / "model_mtp.safetensors")
    quant = json.loads((destination / "config.json").read_text())["quantization_config"]
    group = QuantizationScheme.model_validate(
        next(iter(quant["config_groups"].values()))
    )
    compressor = BaseCompressor.get_value_from_registry(group.format)
    for name, original in expected.items():
        module = name.removesuffix(".weight")
        if f"{module}.weight_scale" in output:
            tensors = {
                param: output[f"{module}.{param}"]
                for param in compressor.compression_param_names(group)
            }
            restored = compressor.decompress(tensors, group)["weight"].float()
            error = (
                restored - original.float()
            ).square().mean() / original.float().square().mean()
            assert error < 0.04, (name, error)
            assert not any(match_name(module, pattern) for pattern in quant["ignore"])
        else:
            # Transformers preserves e_score_correction_bias in fp32.
            torch.testing.assert_close(
                output[name], original, rtol=0, atol=0, check_dtype=False
            )
    assert f"{PREFIX}mlp.experts.0.up_proj.weight_scale" in output
    assert f"{PREFIX}self_attn.indexer.wq_b.weight_scale" in output
    assert not any(name.endswith("weight_scale_inv") for name in output)
    assert any(
        match_name(f"{PREFIX}self_attn.fused_qkv_a_proj", target)
        for target in group.targets
    )
    for projection in ("wk", "weights_proj"):
        assert f"{PREFIX}self_attn.indexer.{projection}" in quant["ignore"]
        assert f"{PREFIX}self_attn.indexer.{projection}.weight_scale" not in output
    weights = get_weight_mappings(destination)
    assert set(output) <= weights.keys()
    assert "model.layers.0.self_attn.q_a_proj.weight_scale" in weights
    assert not any(name.startswith("mtp.") for name in weights)
    if scheme == "NVFP4A16":
        for first, second in (
            ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"),
            ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj"),
        ):
            torch.testing.assert_close(
                output[f"{PREFIX}{first}.weight_global_scale"],
                output[f"{PREFIX}{second}.weight_global_scale"],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "bad_scale",
    [None, torch.ones(1, 2), torch.tensor([[float("nan")]]), torch.zeros(1, 1)],
)
def test_glm53_rejects_invalid_mtp_scales(tmp_path, bad_scale):
    model, _ = _source_model(tmp_path, True)
    checkpoint = tmp_path / "source" / "model.safetensors"
    weights = load_file(checkpoint)
    name = f"{PREFIX}self_attn.q_a_proj.weight_scale_inv"
    if bad_scale is None:
        del weights[name]
    else:
        weights[name] = bad_scale
    save_file(weights, checkpoint)
    with pytest.raises(ValueError, match="GLM-5.3 MTP FP8"):
        load_mtp_model(model)
    assert not hasattr(model, "mtp")


@pytest.mark.parametrize(
    "attribute,value", [("is_quantized", True), ("model_type", "glm5_next")]
)
def test_native_fp8_exception_is_glm53_and_dequantized_only(tmp_path, attribute, value):
    model, _ = _source_model(tmp_path, True)
    setattr(model.config if attribute == "model_type" else model, attribute, value)
    entrypoint = Oneshot.__new__(Oneshot)
    entrypoint.model_args = SimpleNamespace(trust_remote_code_model=False)
    with pytest.raises(ValueError, match="different format"):
        entrypoint.validate_model(model)


def test_glm53_recipe_loads_mtp(tmp_path):
    model, _ = _source_model(tmp_path, True)
    reset_session()
    oneshot(
        model=model,
        recipe=QuantizationModifier(
            scheme={"FP8_DYNAMIC": [r"re:^mtp\.layers\."]}, ignore=IGNORE
        ),
    )
    assert hasattr(model.mtp.layers[0].mtp_block.self_attn.q_a_proj, "weight_scale")
