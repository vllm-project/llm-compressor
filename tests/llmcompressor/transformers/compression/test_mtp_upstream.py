import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import (
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    InklingForCausalLM,
    InklingTextConfig,
    PretrainedConfig,
)
from transformers.monkey_patching import (
    get_patch_mapping,
    register_patch_mapping,
    unregister_patch_mapping,
)

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.mtp import (
    _mtp_weights,
    load_mtp_model,
    save_mtp_tensors,
)

MtpModel = getattr(
    pytest.importorskip("transformers.modeling_layers"), "MtpModel", None
)
if MtpModel is None:
    pytest.skip("Transformers does not provide MtpModel", allow_module_level=True)


def _source_model(tmp_path, with_mtp=True):
    config = InklingTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_head_dim=16,
        intermediate_size=128,
        layer_types=["hybrid", "hybrid"],
        mlp_layer_types=["dense", "dense"],
        num_mtp_layers=1,
        conv_kernel_size=2,
        max_position_embeddings=128,
    )
    model = InklingForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source)
    if with_mtp:
        mtp = MtpModel(model, 1)
        weights = load_file(source / "model.safetensors")
        for name, tensor in mtp.layers[0].state_dict().items():
            name = (
                name.replace("mtp_block.", "transformer_block.")
                .replace("enorm.", "embed_norm.")
                .replace("hnorm.", "hidden_norm.")
                .replace("eh_proj.", "input_proj.")
            )
            weights[f"model.mtp.layers.0.{name}"] = tensor.contiguous()
        save_file(weights, source / "model.safetensors")
    model.config.name_or_path = str(source)
    model.config.dtype = torch.float32
    model.name_or_path = str(source)
    model._weight_conversions = []
    return model


def test_mtp_target_quantizes_with_upstream_model(tmp_path):
    model = _source_model(tmp_path)
    recipe = QuantizationModifier(
        config_groups={
            "mtp": preset_name_to_scheme("FP8_DYNAMIC", [r"re:^mtp\.layers\."]),
            "backbone": preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]),
        },
        ignore=["lm_head"],
    )
    with patch(
        "llmcompressor.transformers.compression.mtp._mtp_weights",
        side_effect=AssertionError("load must not rescan checkpoint metadata"),
    ):
        oneshot(model=model, recipe=recipe)
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    weights = get_weight_mappings(destination)
    mtp_weight = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    assert mtp_weight in weights
    assert mtp_weight.replace(".weight", ".weight_scale") in weights
    assert "model.layers.0.mlp.up_proj.weight_scale" in weights
    assert not any(name.startswith("mtp.") for name in weights)
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert any(
        group["targets"] == [r"re:^model\.mtp\..*"]
        for group in quant["config_groups"].values()
    )
    assert mtp_weight.removesuffix(".weight") not in quant["ignore"]
    assert hasattr(model, "mtp")

    reloaded = InklingForCausalLM.from_pretrained(destination)
    assert len(MtpModel.from_pretrained(reloaded).layers) == 1


def test_mixed_target_group_keeps_backbone_target(tmp_path):
    model = _source_model(tmp_path)
    recipe = QuantizationModifier(
        config_groups={
            "mixed": preset_name_to_scheme(
                "FP8_DYNAMIC", ["Linear", r"re:^mtp\.layers\."]
            )
        },
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    with open(destination / "config.json", encoding="utf-8") as handle:
        groups = json.load(handle)["quantization_config"]["config_groups"]
    targets = next(iter(groups.values()))["targets"]
    assert targets == ["Linear", r"re:^model\.mtp\..*"]


def test_untargeted_mtp_copied_from_checkpoint(tmp_path):
    model = _source_model(tmp_path)
    recipe = QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
    destination = tmp_path / "destination"
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        model.save_pretrained(destination)
    finally:
        logger.remove(handler_id)

    assert any("MTP weights were not targeted" in log for log in logs)
    weights = get_weight_mappings(destination)
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight" in weights
    assert (
        "model.mtp.layers.0.transformer_block.mlp.up_proj.weight_scale" not in weights
    )
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert r"re:^model\.mtp\..*" in quant["ignore"]


def test_unquantized_mtp_copy_from_quantized_backbone_config(tmp_path):
    model = _source_model(tmp_path)
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    model.config.quantization_config = {"quant_method": "fp8"}
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight" in (
        get_weight_mappings(destination)
    )


@pytest.mark.parametrize("source_format", ["scale", "float8"])
def test_source_quantized_mtp_copy_requires_conversion(tmp_path, source_format):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    weights = load_file(source)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    if source_format == "scale":
        weights[f"{name}_scale"] = torch.ones(1)
    else:
        weights[name] = weights[name].to(torch.float8_e4m3fn)
    save_file(weights, source)
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))

    with pytest.raises(ValueError, match="Cannot copy source-quantized MTP weights"):
        model.save_pretrained(tmp_path / "destination")


def test_dense_save_drops_mtp_qparams(tmp_path):
    model = _source_model(tmp_path)
    recipe = QuantizationModifier(scheme={"FP8_DYNAMIC": [r"re:^mtp\.layers\."]})
    oneshot(model=model, recipe=recipe)
    destination = tmp_path / "destination"
    model.save_pretrained(destination, save_compressed=False)

    weights = get_weight_mappings(destination)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    assert name in weights
    assert name.replace(".weight", ".weight_scale") not in weights
    assert load_file(destination / "model_mtp.safetensors")[name].dtype == model.dtype
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert name.removesuffix(".weight") in quant["ignore"]


def test_config_hint_without_mtp_weights_does_not_copy(tmp_path):
    model = _source_model(tmp_path, with_mtp=False)
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    destination = tmp_path / "destination"
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        model.save_pretrained(destination)
    finally:
        logger.remove(handler_id)

    assert any("no MTP checkpoint weights" in log for log in logs)
    assert not any(
        name.startswith("model.mtp.") for name in get_weight_mappings(destination)
    )


@pytest.mark.parametrize("patterns", [[], [r"unexpected.*"]])
def test_non_mtp_config_does_not_require_layer_count(tmp_path, patterns):
    model = SimpleNamespace(
        config=PretrainedConfig(), _keys_to_ignore_on_load_unexpected=patterns
    )
    save_mtp_tensors(model, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_unsupported_mtp_target_points_to_fallback(tmp_path):
    model = _source_model(tmp_path)
    model._keys_to_ignore_on_load_unexpected = []
    recipe = QuantizationModifier(scheme={"FP8_DYNAMIC": [r"re:^mtp\.layers\."]})
    with pytest.raises(ValueError, match="mtp_fp8_fallback.py"):
        oneshot(model=model, recipe=recipe)


def test_missing_mtp_checkpoint_weights_point_to_fallback(tmp_path):
    model = _source_model(tmp_path, with_mtp=False)
    recipe = QuantizationModifier(scheme={"FP8_DYNAMIC": [r"re:^mtp\.layers\."]})
    with pytest.raises(ValueError, match="mtp_fp8_fallback.py"):
        oneshot(model=model, recipe=recipe)


def test_unsupported_untargeted_mtp_copies_with_fallback_warning(tmp_path):
    model = _source_model(tmp_path)
    model._keys_to_ignore_on_load_unexpected = []
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    destination = tmp_path / "destination"
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        model.save_pretrained(destination)
    finally:
        logger.remove(handler_id)

    assert any("mtp_fp8_fallback.py" in log for log in logs)
    assert (
        "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
        in get_weight_mappings(destination)
    )


def test_missing_mtp_weights_point_to_fallback(tmp_path):
    model = _source_model(tmp_path)
    with patch(
        "transformers.modeling_layers.MtpModel.from_pretrained",
        side_effect=RuntimeError("The following MtpModel weights are missing"),
    ):
        with pytest.raises(ValueError, match="mtp_fp8_fallback.py"):
            load_mtp_model(model)

    with patch(
        "transformers.modeling_layers.MtpModel.from_pretrained",
        side_effect=RuntimeError("device failure"),
    ):
        with pytest.raises(RuntimeError, match="device failure"):
            load_mtp_model(model)


def test_mtp_loader_accepts_string_execution_device(tmp_path):
    model = _source_model(tmp_path)
    with (
        patch(
            "llmcompressor.transformers.compression.mtp.get_execution_device",
            return_value="cuda",
        ),
        patch(
            "llmcompressor.transformers.compression.mtp.set_onload_device"
        ) as set_device,
    ):
        load_mtp_model(model)

    assert set_device.call_args.args[1] == torch.device("cuda")


def test_mtp_load_preserves_unrelated_patch_mapping(tmp_path):
    model = _source_model(tmp_path)
    experts_cls = type("MtpTestExperts", (torch.nn.Module,), {})
    register_patch_mapping({"UnrelatedMtpTestClass": torch.nn.Linear})
    try:
        with (
            patch(
                "llmcompressor.transformers.compression.mtp.has_linearize_load_mappings",
                return_value=True,
            ),
            patch(
                "llmcompressor.transformers.compression.mtp.get_linearize_load_mappings",
                return_value=(experts_cls, None, None),
            ),
            patch(
                "llmcompressor.transformers.compression.mtp.LinearExperts2D.get_linear_experts_cls",
                return_value=experts_cls,
            ),
            patch("transformers.modeling_layers.MtpModel.from_pretrained") as load,
        ):
            load.return_value.use_shared_post_norm = False
            load_mtp_model(model)
        assert get_patch_mapping()["UnrelatedMtpTestClass"] is torch.nn.Linear
        assert experts_cls.__name__ not in get_patch_mapping()
    finally:
        unregister_patch_mapping(["UnrelatedMtpTestClass"])


@pytest.mark.parametrize(
    "error", [AttributeError("bad attribute"), ValueError("invalid config")]
)
def test_unexpected_mtp_load_error_is_not_reframed(tmp_path, error):
    model = _source_model(tmp_path)
    with patch(
        "transformers.modeling_layers.MtpModel.from_pretrained", side_effect=error
    ):
        with pytest.raises(type(error), match=str(error)):
            load_mtp_model(model)


def test_calibrated_mtp_target_is_not_silently_skipped(tmp_path):
    model = _source_model(tmp_path)
    recipe = QuantizationModifier(scheme={"NVFP4": [r"re:^mtp\.layers\."]})
    with pytest.raises(ValueError, match="data-free schemes only"):
        oneshot(model=model, recipe=recipe)


@pytest.mark.parametrize("target_mtp", [True, False])
def test_trailing_mtp_layer_uses_upstream_weight_mapping(tmp_path, target_mtp):
    config = Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        num_nextn_predict_layers=1,
        max_position_embeddings=128,
    )
    model = Glm4MoeForCausalLM(config).eval()
    mtp = MtpModel(model, 1)
    source = tmp_path / "source"
    model.save_pretrained(source)
    weights = load_file(source / "model.safetensors")
    for name, tensor in mtp.layers[0].state_dict().items():
        name = name.removeprefix("mtp_block.")
        if name == "post_norm.weight":
            name = "shared_head.norm.weight"
        weights[f"model.layers.2.{name}"] = tensor.contiguous()
    save_file(weights, source / "model.safetensors")
    model.config.name_or_path = str(source)
    model.config.dtype = torch.float32
    model.name_or_path = str(source)
    model._weight_conversions = []
    model._keys_to_ignore_on_load_unexpected = []
    assert "model.layers.2.eh_proj.weight" in _mtp_weights(model)[0]
    model._keys_to_ignore_on_load_unexpected = [r"model\.layers\.2.*"]
    linearize_moe(model)

    from_pretrained = MtpModel.from_pretrained

    def load_without_patch(main_model):
        loaded = from_pretrained(main_model)
        linearize_moe(loaded)
        return loaded

    if target_mtp:
        recipe = QuantizationModifier(
            config_groups={
                "mtp": preset_name_to_scheme("FP8_DYNAMIC", [r"re:^mtp\.layers\."]),
                "backbone": preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]),
            },
            ignore=["lm_head"],
        )
    else:
        recipe = QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"])
    with (
        patch(
            "transformers.modeling_layers.MtpModel.from_pretrained",
            side_effect=load_without_patch,
        ),
        patch(
            "llmcompressor.transformers.compression.mtp.has_linearize_load_mappings",
            return_value=False,
        ),
    ):
        oneshot(model=model, recipe=recipe)

    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    weights = get_weight_mappings(destination)
    if target_mtp:
        assert "model.layers.2.self_attn.q_proj.weight_scale" in weights
        assert "model.layers.2.mlp.experts.0.up_proj.weight_scale" in weights
    else:
        assert "model.layers.2.self_attn.q_proj.weight" in weights
        assert "model.layers.2.self_attn.q_proj.weight_scale" not in weights
    assert not any(name.startswith("mtp.") for name in weights)
