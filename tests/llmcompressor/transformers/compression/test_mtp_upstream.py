import json
import shutil
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload import OffloadCache
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from loguru import logger
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import (
    DeepseekV3ForCausalLM,
    Glm4MoeForCausalLM,
    Glm4MoeLiteForCausalLM,
    GlmMoeDsaForCausalLM,
    InklingForCausalLM,
    InklingTextConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.transformers.compression.mtp import (
    _mtp_weights,
    load_with_mtp_model,
    save_mtp_tensors,
)
from llmcompressor.utils import load_context

MtpModel = getattr(
    pytest.importorskip("transformers.modeling_layers"), "MtpModel", None
)
if MtpModel is None:
    pytest.skip("Transformers does not provide MtpModel", allow_module_level=True)


def _source_model(tmp_path, with_mtp=True, load_mtp=False):
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
        mtp.layers.apply(model._init_weights)
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
    if load_mtp:
        with load_context(InklingForCausalLM, load_mtp=True):
            return InklingForCausalLM.from_pretrained(source, local_files_only=True)
    return model


def test_mtp_target_quantizes_with_upstream_model(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
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
        mtp_weight.removesuffix(".weight") in group["targets"]
        for group in quant["config_groups"].values()
    )
    assert mtp_weight.removesuffix(".weight") not in quant["ignore"]
    assert hasattr(model, "mtp")

    reloaded = InklingForCausalLM.from_pretrained(destination)
    assert len(MtpModel.from_pretrained(reloaded).layers) == 1


def test_mtp_participates_in_initial_offload_and_sharded_save(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    assert isinstance(model.mtp.layers[0].eh_proj._parameters, OffloadCache)
    assert model.mtp.embed_tokens is model.get_input_embeddings()
    assert model.mtp.shared_head is model.lm_head
    assert "embed_tokens" not in model.mtp._modules
    reference = model.mtp.layers[0].mtp_block.mlp.up_proj.weight.detach().clone()
    oneshot(
        model=model,
        recipe=QuantizationModifier(
            scheme="FP8_DYNAMIC", ignore=["lm_head", r"re:.*\.eh_proj$"]
        ),
    )
    for directory in ("first", "second"):
        destination = tmp_path / directory
        model.save_pretrained(destination, max_shard_size="20KB")
        assert not (destination / "model_mtp.safetensors").exists()
        mapping = get_weight_mappings(destination)
        assert len(set(mapping.values())) > 1
        name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
        tensor = load_file(mapping[name])[name].float()
        scale = load_file(mapping[name + "_scale"])[name + "_scale"]
        torch.testing.assert_close(tensor * scale, reference, atol=0.005, rtol=0.15)
        assert model.mtp.shared_head is model.lm_head
        assert model.mtp.embed_tokens is model.get_input_embeddings()


def test_mtp_exists_before_non_source_rank_sync(tmp_path):
    model = _source_model(tmp_path)
    seen = []

    def sync(loaded):
        assert hasattr(loaded, "mtp")
        assert all(param.is_meta for param in loaded.parameters())
        seen.append(loaded)

    with (
        patch("compressed_tensors.offload.load.is_source_process", return_value=False),
        patch("compressed_tensors.offload.load.from_accelerate", side_effect=sync),
        load_context(InklingForCausalLM, load_mtp=True),
    ):
        loaded = InklingForCausalLM.from_pretrained(model.name_or_path)
    assert seen == [loaded]


def test_mtp_loader_resolves_pinned_cached_snapshot(tmp_path):
    model = _source_model(tmp_path)
    source = model.name_or_path
    original = InklingForCausalLM.from_pretrained

    def load_backbone(*args, **kwargs):
        loaded = original(source)
        loaded.config.name_or_path = "example/model"
        loaded.config._commit_hash = "pinned-commit"
        return loaded

    with (
        patch.object(InklingForCausalLM, "from_pretrained", side_effect=load_backbone),
        patch(
            "llmcompressor.transformers.compression.mtp.cached_file",
            return_value=str(tmp_path / "source" / "config.json"),
        ) as resolve,
        load_with_mtp_model(InklingForCausalLM),
    ):
        loaded = InklingForCausalLM.from_pretrained(
            "example/model",
            cache_dir="custom-cache",
            local_files_only=True,
            revision="requested-branch",
            token="test-token",
            subfolder="weights",
        )
    resolve.assert_called_once_with(
        "example/model",
        "config.json",
        revision="pinned-commit",
        cache_dir="custom-cache",
        local_files_only=True,
        token="test-token",
        subfolder="weights",
    )
    assert loaded.config.name_or_path == "example/model"
    assert hasattr(loaded, "mtp")


def test_mtp_source_quantization_fails_during_loading(tmp_path):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    weights = load_file(source)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    weights[name] = weights[name].to(torch.float8_e4m3fn)
    weights[name + "_scale"] = torch.ones(128, 1)
    save_file(weights, source)
    with load_context(InklingForCausalLM, load_mtp=True):
        with pytest.raises(ValueError, match="source-quantized MTP"):
            InklingForCausalLM.from_pretrained(model.name_or_path)


@pytest.mark.parametrize(
    "targets",
    [
        ["Linear", r"re:^mtp\.layers\."],
        [r"re:^(?:model\.layers|mtp\.layers)\."],
    ],
)
def test_mixed_target_group_keeps_backbone_target(tmp_path, targets):
    model = _source_model(tmp_path, load_mtp=True)
    recipe = QuantizationModifier(
        config_groups={"mixed": preset_name_to_scheme("FP8_DYNAMIC", targets)},
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    with open(destination / "config.json", encoding="utf-8") as handle:
        groups = json.load(handle)["quantization_config"]["config_groups"]
    saved_targets = next(iter(groups.values()))["targets"]
    assert all(target in saved_targets for target in targets)
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj" in saved_targets
    weights = get_weight_mappings(destination)
    assert "model.layers.0.mlp.up_proj.weight_scale" in weights
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight_scale" in weights


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
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj" in quant["ignore"]


def test_mtp_copy_uses_backbone_hub_revision(tmp_path):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    model.name_or_path = "example/model"
    model.config._commit_hash = "pinned-commit"
    with patch(
        "llmcompressor.transformers.compression.mtp.cached_file",
        side_effect=[None, str(source)],
    ) as lookup:
        save_mtp_tensors(model, str(destination))

    assert len(lookup.call_args_list) == 2
    assert all(
        call.kwargs["revision"] == "pinned-commit" for call in lookup.call_args_list
    )


@pytest.mark.parametrize("subfolder", ["", "weights"])
@pytest.mark.parametrize("use_load_context", [True, False])
def test_mtp_copy_from_custom_offline_cache(tmp_path, subfolder, use_load_context):
    source = _source_model(tmp_path)
    cache = tmp_path / "custom-cache"
    revision = "a" * 40
    snapshot = cache / "models--example--mtp" / "snapshots" / revision / subfolder
    shutil.copytree(source.name_or_path, snapshot)
    reference = load_file(snapshot / "model.safetensors")
    context = load_context(InklingForCausalLM) if use_load_context else nullcontext()
    with (
        patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
        patch("transformers.utils.hub.is_offline_mode", return_value=True),
        context,
    ):
        model = InklingForCausalLM.from_pretrained(
            "example/mtp",
            cache_dir=cache,
            revision=revision,
            subfolder=subfolder,
            local_files_only=True,
        )
        assert model.name_or_path == "example/mtp"
        assert not hasattr(model, "mtp")
        if use_load_context:
            oneshot(
                model=model,
                recipe=QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"]),
            )
        else:
            modify_save_pretrained(model)
        destination = tmp_path / "destination"
        kwargs = {} if use_load_context else {"mtp_source": str(snapshot)}
        model.save_pretrained(destination, **kwargs)
        assert model.name_or_path == "example/mtp"

    weights = get_weight_mappings(destination)
    for name, tensor in reference.items():
        if name.startswith("model.mtp."):
            torch.testing.assert_close(
                load_file(weights[name])[name], tensor, rtol=0, atol=0
            )


def test_unquantized_mtp_copy_from_quantized_backbone_config(tmp_path):
    model = _source_model(tmp_path)
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    model.config.quantization_config = {"quant_method": "fp8"}
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight" in (
        get_weight_mappings(destination)
    )


@pytest.mark.parametrize("source_format", ["scale", "float8", "packed_fp4"])
def test_source_quantized_mtp_copy_requires_conversion(tmp_path, source_format):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    weights = load_file(source)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    if source_format == "scale":
        weights[f"{name}_scale"] = torch.ones(1)
    elif source_format == "float8":
        weights[name] = weights[name].to(torch.float8_e4m3fn)
    else:
        weights[name] = torch.zeros_like(weights[name], dtype=torch.int8)
        weights[f"{name.removesuffix('.weight')}.scale"] = torch.ones(1)
    save_file(weights, source)
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))

    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="Cannot copy source-quantized MTP weights"):
        model.save_pretrained(destination)
    assert not destination.exists()


def test_dense_save_uses_normal_model_serialization(tmp_path):
    model = _source_model(tmp_path, load_mtp=True).to(torch.bfloat16)
    recipe = QuantizationModifier(scheme={"FP8_DYNAMIC": [r"re:^mtp\.layers\."]})
    oneshot(model=model, recipe=recipe)
    model.mtp.to(torch.bfloat16)
    model.mtp.layers[0].register_buffer(
        "e_score_correction_bias", torch.ones(2, dtype=torch.float32)
    )
    destination = tmp_path / "destination"
    model.save_pretrained(destination, save_compressed=False)

    weights = get_weight_mappings(destination)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    assert name in weights
    assert name.replace(".weight", ".weight_scale") in weights
    tensors = load_file(weights[name])
    assert tensors[name].dtype == model.dtype
    assert tensors["model.mtp.layers.0.e_score_correction_bias"].dtype == torch.float32
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert quant["format"] == "dense"
    assert not (destination / "model_mtp.safetensors").exists()


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


def test_mtp_targets_require_loaded_mtp(tmp_path):
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


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("bad attribute"),
        ValueError("invalid config"),
        RuntimeError("device failure"),
        RuntimeError("MTP weights are missing"),
    ],
)
def test_unexpected_mtp_load_error_is_not_reframed(tmp_path, error):
    model = _source_model(tmp_path)
    with patch(
        "transformers.modeling_layers.MtpModel.from_pretrained", side_effect=error
    ):
        with pytest.raises(type(error), match=str(error)):
            with load_context(InklingForCausalLM, load_mtp=True):
                InklingForCausalLM.from_pretrained(model.name_or_path)


def test_calibrated_mtp_target_is_not_silently_skipped(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    recipe = QuantizationModifier(scheme={"NVFP4": [r"re:^mtp\.layers\."]})
    with pytest.raises(ValueError, match="data-free schemes only"):
        oneshot(model=model, recipe=recipe)


@pytest.mark.parametrize("mtp_first", [True, False])
def test_calibrated_backbone_with_data_free_mtp(tmp_path, mtp_first):
    model = _source_model(tmp_path, load_mtp=True)
    backbone = QuantizationModifier(
        scheme="NVFP4", targets=["Linear"], ignore=["lm_head", r"re:^mtp\."]
    )
    mtp = QuantizationModifier(
        scheme="FP8_DYNAMIC",
        targets=[r"re:^mtp\.layers\."],
        ignore=[r"re:.*\.eh_proj$"],
    )
    recipe = [mtp, backbone] if mtp_first else [backbone, mtp]
    dataset = DataLoader(
        [
            {
                "input_ids": torch.randint(0, 128, (16,)),
                "attention_mask": torch.ones(16, dtype=torch.long),
            }
            for _ in range(2)
        ]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    oneshot(
        model=model,
        recipe=recipe,
        dataset=dataset,
        processor=tokenizer,
        pipeline="independent",
        sequential_targets=[r"re:^model\.layers\.\d+$"],
    )
    destination = tmp_path / "destination"
    model.save_pretrained(destination)

    weights = get_weight_mappings(destination)
    backbone_prefix = "model.layers.0.mlp.up_proj"
    mtp_prefix = "model.mtp.layers.0.transformer_block.mlp.up_proj"
    backbone_tensors = load_file(weights[f"{backbone_prefix}.weight_packed"])
    mtp_tensors = load_file(weights[f"{mtp_prefix}.weight"])
    assert backbone_tensors[f"{backbone_prefix}.weight_packed"].dtype == torch.uint8
    assert mtp_tensors[f"{mtp_prefix}.weight"].dtype == torch.float8_e4m3fn
    for name in (f"{backbone_prefix}.input_global_scale", f"{mtp_prefix}.weight_scale"):
        scale = load_file(weights[name])[name]
        assert torch.isfinite(scale).all() and (scale > 0).all()
    assert f"{mtp_prefix}.input_global_scale" not in weights
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert quant["format"] == "mixed-precision"
    assert {
        group["weights"]["num_bits"] for group in quant["config_groups"].values()
    } == {4, 8}


def _source_glm_model(tmp_path, model_cls=Glm4MoeForCausalLM):
    # Keep the released depth because upstream MtpModel discovers MTP using
    # architecture-defined layer indices. Shrink only widths and expert counts.
    num_layers = {
        Glm4MoeForCausalLM: 46,
        Glm4MoeLiteForCausalLM: 47,
        GlmMoeDsaForCausalLM: 78,
        DeepseekV3ForCausalLM: 61,
    }[model_cls]
    config = model_cls.config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=num_layers,
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
        num_mtp_layers=1,
        max_position_embeddings=128,
    )
    if getattr(config, "layer_types", None) is not None:
        config.mtp_layer_types = [config.layer_types[-1]]
    if getattr(config, "mlp_layer_types", None) is not None:
        config.mtp_mlp_layer_types = [config.mlp_layer_types[-1]]
    model = model_cls(config).eval()
    mtp = MtpModel(model, 1)
    # MtpModel's generic initializer does not initialize architecture-specific
    # fused expert tensors, which are allocated with torch.empty.
    mtp.layers.apply(model._init_weights)
    linearize_moe(mtp)
    source = tmp_path / "source"
    model.save_pretrained(source)
    weights = load_file(source / "model.safetensors")
    for name, tensor in mtp.layers[0].state_dict().items():
        name = name.removeprefix("mtp_block.")
        if name == "post_norm.weight":
            name = "shared_head.norm.weight"
        weights[f"model.layers.{num_layers}.{name}"] = tensor.contiguous()
    save_file(weights, source / "model.safetensors")
    model.config.name_or_path = str(source)
    model.config.dtype = torch.float32
    model.name_or_path = str(source)
    model._weight_conversions = []
    assert f"model.layers.{num_layers}.eh_proj.weight" in _mtp_weights(model)[0]
    return model


@pytest.mark.parametrize(
    "model_cls",
    [
        Glm4MoeForCausalLM,
        Glm4MoeLiteForCausalLM,
        GlmMoeDsaForCausalLM,
        DeepseekV3ForCausalLM,
    ],
)
@pytest.mark.parametrize("target_mtp", [True, False])
def test_trailing_mtp_layer_uses_upstream_weight_mapping(
    tmp_path, target_mtp, model_cls
):
    model = _source_glm_model(tmp_path, model_cls)
    source = model.name_or_path
    prefix = f"model.layers.{model.config.num_hidden_layers}."
    with load_context(model_cls, load_mtp=target_mtp):
        model = model_cls.from_pretrained(source)

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
    oneshot(model=model, recipe=recipe)

    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    weights = get_weight_mappings(destination)
    if target_mtp:
        assert any(
            name.startswith(f"{prefix}self_attn.") and name.endswith("weight_scale")
            for name in weights
        )
        assert f"{prefix}mlp.experts.0.up_proj.weight_scale" in weights
    else:
        assert any(
            name.startswith(f"{prefix}self_attn.") and name.endswith(".weight")
            for name in weights
        )
        assert not any(
            name.startswith(prefix) and name.endswith("weight_scale")
            for name in weights
        )
    assert not any(name.startswith("mtp.") for name in weights)
