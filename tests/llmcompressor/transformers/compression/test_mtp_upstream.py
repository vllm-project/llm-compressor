import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload import OffloadCache, get_device_map
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from datasets import Dataset
from loguru import logger
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3ForCausalLM,
    Glm4MoeForCausalLM,
    Glm4MoeLiteForCausalLM,
    GlmMoeDsaForCausalLM,
    InklingForCausalLM,
    InklingTextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Qwen3_5MoeConfig,
)

from llmcompressor import oneshot, reset_session
from llmcompressor.entrypoints.oneshot import _calibrated_names
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.transformers.compression.mtp import (
    _mtp_weights,
    mtp_calibration_context,
    save_mtp_tensors,
    validate_mtp_copy_source,
)
from llmcompressor.utils import load_context
from llmcompressor.utils.pytorch.module import get_no_split_params

MtpModel = getattr(
    pytest.importorskip("transformers.modeling_layers"), "MtpModel", None
)
if MtpModel is None:
    pytest.skip("Transformers does not provide MtpModel", allow_module_level=True)


def _source_model(tmp_path, with_mtp=True, load_mtp=False, num_mtp_layers=1):
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
        num_mtp_layers=num_mtp_layers,
        conv_kernel_size=2,
        max_position_embeddings=128,
    )
    model = InklingForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source)
    if with_mtp:
        mtp = MtpModel(model, num_mtp_layers)
        mtp.layers.apply(model._init_weights)
        weights = load_file(source / "model.safetensors")
        for index, layer in enumerate(mtp.layers):
            for name, tensor in layer.state_dict().items():
                name = (
                    name.replace("mtp_block.", "transformer_block.")
                    .replace("enorm.", "embed_norm.")
                    .replace("hnorm.", "hidden_norm.")
                    .replace("eh_proj.", "input_proj.")
                )
                weights[f"model.mtp.layers.{index}.{name}"] = tensor.contiguous()
        save_file(weights, source / "model.safetensors")
    model.config.name_or_path = str(source)
    model.config.dtype = torch.float32
    model.name_or_path = str(source)
    model._weight_conversions = []
    if load_mtp:
        with load_context(InklingForCausalLM):
            return InklingForCausalLM.from_pretrained(source, local_files_only=True)
    return model


def test_load_context_without_mtp(tmp_path):
    source = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    source.save_pretrained(tmp_path)
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)

    assert not hasattr(model, "mtp")
    torch.testing.assert_close(model.state_dict(), source.state_dict())


def test_mtp_quantizes_with_backbone_using_upstream_model(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    recipe = QuantizationModifier(
        scheme="FP8_DYNAMIC",
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
    assert all(
        group["targets"] == ["Linear"] for group in quant["config_groups"].values()
    )
    assert mtp_weight.removesuffix(".weight") not in quant["ignore"]
    assert hasattr(model, "mtp")

    reloaded = InklingForCausalLM.from_pretrained(destination)
    assert len(MtpModel.from_pretrained(reloaded).layers) == 1


def test_mtp_participates_in_initial_offload_and_sharded_save(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    assert isinstance(model.mtp.layers[0].eh_proj._parameters, OffloadCache)
    embedding = model.mtp.embed_tokens
    if embedding is not model.get_input_embeddings():
        assert isinstance(embedding.embed_norm, torch.nn.Identity)
    assert isinstance(embedding._parameters, OffloadCache)
    torch.testing.assert_close(
        embedding.weight, model.get_input_embeddings().weight, rtol=0, atol=0
    )
    assert model.mtp.shared_head is model.lm_head
    assert (model.mtp._modules.get("embed_tokens") is embedding) == (
        embedding is not model.get_input_embeddings()
    )
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
        assert "model.mtp.embed_tokens.weight" not in mapping
        assert len(set(mapping.values())) > 1
        name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
        tensor = load_file(mapping[name])[name].float()
        scale = load_file(mapping[name + "_scale"])[name + "_scale"]
        torch.testing.assert_close(tensor * scale, reference, atol=0.005, rtol=0.15)
        assert model.mtp.shared_head is model.lm_head
        assert model.mtp.embed_tokens is embedding
        assert isinstance(embedding._parameters, OffloadCache)


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


@pytest.mark.parametrize("unsupported_config", [False, True])
def test_mtp_source_quantization_fails_during_loading(
    tmp_path, mocker, unsupported_config
):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    weights = load_file(source)
    name = "model.mtp.layers.0.transformer_block.mlp.up_proj.weight"
    weights[name] = weights[name].to(torch.float8_e4m3fn)
    weights[name + "_scale"] = torch.ones(128, 1)
    save_file(weights, source)
    if unsupported_config:
        mocker.patch.object(
            InklingTextConfig, "get_mtp_config", side_effect=ValueError("unsupported")
        )
    with load_context(InklingForCausalLM, load_mtp=True):
        with pytest.raises(ValueError, match="source-quantized MTP"):
            InklingForCausalLM.from_pretrained(model.name_or_path)


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

    assert not any("MTP" in log for log in logs)
    weights = get_weight_mappings(destination)
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight" in weights
    assert (
        "model.mtp.layers.0.transformer_block.mlp.up_proj.weight_scale" not in weights
    )
    with open(destination / "config.json", encoding="utf-8") as handle:
        quant = json.load(handle)["quantization_config"]
    assert any(
        match_name("model.mtp.layers.0.transformer_block.mlp.up_proj", target)
        for target in quant["ignore"]
    )


@pytest.mark.parametrize("entrypoint", ["helper", "save"])
def test_mtp_copy_uses_backbone_hub_revision(tmp_path, entrypoint):
    model = _source_model(tmp_path)
    source = tmp_path / "source" / "model.safetensors"
    destination = tmp_path / "destination"
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    if entrypoint == "helper":
        model.save_pretrained(destination)
    model.name_or_path = "example/model"
    model.config._commit_hash = "pinned-commit"
    with (
        patch(
            "llmcompressor.transformers.compression.mtp.cached_file",
            side_effect=[None, str(source)],
        ) as lookup,
        patch(
            "llmcompressor.transformers.compression.compressed_tensors_utils."
            "infer_recipe_from_model_path",
            return_value=None,
        ),
        patch(
            "llmcompressor.transformers.compression.compressed_tensors_utils."
            "copy_python_files_from_model_cache",
        ),
    ):
        if entrypoint == "helper":
            source_mtp = validate_mtp_copy_source(model)
            save_mtp_tensors(model, str(destination), source_mtp=source_mtp)
        else:
            model.save_pretrained(destination)

    assert len(lookup.call_args_list) == 2
    assert all(
        call.kwargs["revision"] == "pinned-commit" for call in lookup.call_args_list
    )
    assert "model.mtp.layers.0.transformer_block.mlp.up_proj.weight" in (
        get_weight_mappings(destination)
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
@pytest.mark.parametrize("entrypoint", ["oneshot", "save"])
def test_source_quantized_mtp_copy_requires_conversion(
    tmp_path, source_format, entrypoint
):
    model = _source_model(tmp_path)
    if entrypoint == "save":
        oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
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
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="Cannot copy source-quantized MTP weights"):
        if entrypoint == "oneshot":
            oneshot(
                model=model,
                recipe=QuantizationModifier(scheme="FP8_DYNAMIC"),
                output_dir=str(destination),
            )
        else:
            model.save_pretrained(destination)
    assert not destination.exists()


@pytest.mark.parametrize("target_backbone", [False, True, "int4", "fp8"])
@pytest.mark.parametrize(
    "model_cls",
    [
        InklingForCausalLM,
        Glm4MoeForCausalLM,
        Glm4MoeLiteForCausalLM,
        GlmMoeDsaForCausalLM,
        DeepseekV3ForCausalLM,
    ],
)
def test_named_mtp_quantization_saves_checkpoint_weights(
    tmp_path, target_backbone, model_cls
):
    if model_cls is InklingForCausalLM:
        model = _source_model(tmp_path, load_mtp=True)
        projection = "model.mtp.layers.0.input_proj"
    else:
        source = _source_glm_model(tmp_path, model_cls)
        with load_context(model_cls, load_mtp=True):
            model = model_cls.from_pretrained(source.name_or_path)
        projection = f"model.layers.{model.config.num_hidden_layers}.eh_proj"
    backbone_target = (
        "model.layers.0.mlp.up_proj"
        if model_cls is InklingForCausalLM
        else "model.layers.0.mlp.experts.0.up_proj"
    )
    targets = [r"re:^mtp\.layers\."]
    if target_backbone in ("int4", "fp8"):
        backbone_bits, mtp_bits = (4, 8) if target_backbone == "int4" else (8, 4)
        recipe = QuantizationModifier(
            config_groups={
                name: QuantizationScheme(
                    targets=scheme_targets,
                    weights=QuantizationArgs(
                        num_bits=4, strategy="group", group_size=16
                    ),
                )
                if bits == 4
                else preset_name_to_scheme("FP8_DYNAMIC", targets=scheme_targets)
                for name, bits, scheme_targets in (
                    ("backbone", backbone_bits, [r"re:^model\.layers\."]),
                    ("mtp", mtp_bits, targets),
                )
            },
            ignore=["lm_head"],
        )
    else:
        if target_backbone:
            targets.append(backbone_target)
        recipe = QuantizationModifier(scheme="FP8_DYNAMIC", targets=targets)
    oneshot(model=model, recipe=recipe)
    expected = {}
    original_targets = {}
    original_schemes = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.startswith("mtp."):
            bits = mtp_bits if target_backbone in ("int4", "fp8") else 8
        elif name.startswith("model.layers.") and target_backbone in ("int4", "fp8"):
            bits = backbone_bits
        elif target_backbone is True and name == backbone_target:
            bits = 8
        else:
            assert not hasattr(module, "quantization_scheme"), name
            continue
        assert module.quantization_scheme.weights.num_bits == bits, name
        original_targets[name] = list(module.quantization_scheme.targets)
        original_schemes[tuple(original_targets[name])] = bits
        # These are the checkpoint names written by the source fixtures above.
        if name.startswith("mtp.layers.0."):
            name = name.removeprefix("mtp.layers.0.")
            if model_cls is InklingForCausalLM:
                name = name.replace("mtp_block.", "transformer_block.").replace(
                    "eh_proj", "input_proj"
                )
            else:
                name = name.removeprefix("mtp_block.")
            name = f"{projection.rpartition('.')[0]}.{name}"
        expected[name] = bits

    for name in ("first", "second"):
        destination = tmp_path / name
        model.save_pretrained(destination)
        weights = get_weight_mappings(destination)
        tensors = {
            name: tensor
            for shard in set(weights.values())
            for name, tensor in load_file(shard).items()
        }
        assert not any(name.startswith("mtp.") for name in weights)
        assert {
            name.removesuffix(".weight_scale")
            for name in weights
            if name.endswith(".weight_scale")
        } == set(expected)
        assert projection in expected
        for name, bits in expected.items():
            weight_name = f"{name}.weight_packed" if bits == 4 else f"{name}.weight"
            dtype = torch.int32 if bits == 4 else torch.float8_e4m3fn
            assert tensors[weight_name].dtype == dtype, name
        config = json.loads((destination / "config.json").read_text())
        quantization = config["quantization_config"]
        # Target remapping is deferred: config targets retain the HF names even
        # when the tensors above are saved under different checkpoint names.
        assert {
            tuple(group["targets"]): group["weights"]["num_bits"]
            for group in quantization["config_groups"].values()
        } == original_schemes
        for group in quantization["config_groups"].values():
            expected_format = (
                "pack-quantized"
                if group["weights"]["num_bits"] == 4
                else "float-quantized"
            )
            assert group["format"] == expected_format
        for name, targets in original_targets.items():
            assert model.get_submodule(name).quantization_scheme.targets == targets


def test_dense_save_uses_normal_model_serialization(tmp_path):
    model = _source_model(tmp_path, load_mtp=True).to(torch.bfloat16)
    recipe = QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"])
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
    source_mtp = validate_mtp_copy_source(model)
    save_mtp_tensors(model, str(tmp_path), source_mtp=source_mtp)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["patterns", "num_mtp_layers", "mtp_layer_types"])
def test_unsupported_mtp_copies_through_sharded_save(tmp_path, mocker, missing):
    model = _source_model(tmp_path)
    if missing == "patterns":
        mocker.patch(
            "llmcompressor.transformers.compression.mtp._mtp_patterns", return_value=[]
        )
    else:
        config = Qwen3_5MoeConfig(text_config={"mtp_num_hidden_layers": 1})
        if missing == "mtp_layer_types":
            config.text_config.num_mtp_layers = 1
        with pytest.raises(ValueError, match=missing):
            config.get_mtp_config()
        # Reuse checkpoint tensors with Transformers' actual Qwen config check.
        mocker.patch.object(
            InklingTextConfig, "get_mtp_config", side_effect=config.get_mtp_config
        )
    mocker.patch.object(MtpModel, "from_pretrained", side_effect=AssertionError)
    mocker.patch("llmcompressor.model_free_ptq", side_effect=AssertionError)
    with load_context(InklingForCausalLM):
        model = InklingForCausalLM.from_pretrained(model.name_or_path)
    assert not hasattr(model, "mtp")
    oneshot(model=model, recipe=QuantizationModifier(scheme="FP8_DYNAMIC"))
    destination = tmp_path / "destination"
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        model.save_pretrained(destination, max_shard_size="20KB")
    finally:
        logger.remove(handler_id)

    assert not any("MTP" in log for log in logs)
    mapping = get_weight_mappings(destination)
    assert len(set(mapping.values())) > 2
    mtp = {
        name: tensor
        for name, tensor in load_file(tmp_path / "source" / "model.safetensors").items()
        if name.startswith("model.mtp.")
    }
    assert {name for name in mapping if name.startswith("model.mtp.")} == set(mtp)
    assert any(name.endswith("q_proj.weight_scale") for name in mapping)
    copied = load_file(destination / "model_mtp.safetensors")
    torch.testing.assert_close(copied, mtp, rtol=0, atol=0)
    quant = json.loads((destination / "config.json").read_text())["quantization_config"]
    assert all(
        name.removesuffix(".weight") in quant["ignore"]
        for name in mtp
        if name.endswith(".weight")
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
    original_from_pretrained = InklingForCausalLM.from_pretrained
    with patch(
        "transformers.modeling_layers.MtpModel.from_pretrained", side_effect=error
    ):
        with pytest.raises(type(error), match=str(error)) as exc_info:
            with load_context(InklingForCausalLM, load_mtp=True):
                InklingForCausalLM.from_pretrained(model.name_or_path)
    assert exc_info.value is error
    assert InklingForCausalLM.from_pretrained == original_from_pretrained


def _calibration_dataset(num_samples=4, seq_len=16):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 128, (num_samples, seq_len), generator=generator)
    return Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": torch.ones_like(input_ids).tolist(),
        }
    )


def _tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    vocab.update({f"token{i}": i for i in range(4, 128)})
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )


def _loaded_mtp_model(tmp_path, layout, num_mtp_layers=1):
    if layout == "dedicated":
        return _source_model(tmp_path, load_mtp=True, num_mtp_layers=num_mtp_layers)
    assert num_mtp_layers == 1, "GLM checkpoints declare one MTP layer"
    source = _source_glm_model(tmp_path).name_or_path
    with load_context(Glm4MoeForCausalLM):
        return Glm4MoeForCausalLM.from_pretrained(source)


@pytest.mark.parametrize("layout", ["dedicated", "trailing"])
@pytest.mark.parametrize("pipeline", ["sequential", "basic"])
def test_calibrated_mtp_target_runs_observers(tmp_path, layout, pipeline):
    model = _loaded_mtp_model(tmp_path, layout)
    calls = []
    model.mtp.layers[0].eh_proj.register_forward_hook(lambda *_: calls.append(1))

    recipe = QuantizationModifier(scheme={"NVFP4": [r"re:^mtp\.layers\..*proj$"]})
    oneshot(
        model=model,
        recipe=recipe,
        processor=_tokenizer(),
        dataset=_calibration_dataset(),
        num_calibration_samples=4,
        pipeline=pipeline,
    )

    assert calls
    linears = {
        name: module
        for name, module in model.mtp.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith("proj")
    }
    assert linears
    for name, module in linears.items():
        scale = module.input_global_scale
        assert torch.isfinite(scale).all() and (scale > 0).all(), name

    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    weights = get_weight_mappings(destination)
    mtp_prefixes = ("model.mtp.", f"model.layers.{model.config.num_hidden_layers}.")
    mtp_scales = [
        name
        for name in weights
        if name.endswith(".input_global_scale") and name.startswith(mtp_prefixes)
    ]
    assert len(mtp_scales) == len(linears)
    quant = json.loads((destination / "config.json").read_text())["quantization_config"]
    assert not any(
        name.removesuffix(".input_global_scale") in quant["ignore"]
        for name in mtp_scales
    )


def test_mtp_calibration_does_not_change_backbone_outputs(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    input_ids = torch.tensor(_calibration_dataset(num_samples=1)["input_ids"])
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
        with mtp_calibration_context(model):
            actual = model(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected)


def _mtp_layer_outputs(model, **inputs):
    """Hidden states produced by each MTP layer during a calibration forward pass"""
    mtp = model.mtp
    outputs = []

    def capture(_module, _args, output):
        if mtp.use_shared_post_norm:
            output = mtp.shared_post_norm(output)
        outputs.append(output)

    handles = [layer.register_forward_hook(capture) for layer in mtp.layers]
    try:
        with torch.no_grad(), mtp_calibration_context(model):
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


@pytest.mark.parametrize("layout,num_mtp_layers", [("dedicated", 2), ("trailing", 1)])
def test_mtp_calibration_matches_upstream_layer_forward(
    tmp_path, layout, num_mtp_layers
):
    model = _loaded_mtp_model(tmp_path, layout, num_mtp_layers)
    if not hasattr(model.mtp, "forward_layer"):
        pytest.skip("Transformers does not provide MtpModel.forward_layer")
    input_ids = torch.tensor(_calibration_dataset(num_samples=2)["input_ids"])
    actual = _mtp_layer_outputs(model, input_ids=input_ids)
    assert len(actual) == num_mtp_layers

    # Teacher forcing: layer `i` embeds the tokens shifted by `i + 1`
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    with torch.no_grad():
        hidden = model.get_decoder()(input_ids=input_ids).last_hidden_state
        for index, output in enumerate(actual):
            shift = index + 1
            hidden = model.mtp.forward_layer(
                index,
                hidden[:, :-1],
                positions[:, shift:],
                None,
                input_ids=input_ids[:, shift:],
            )
            torch.testing.assert_close(output, hidden)


def test_mtp_calibration_masks_left_padding(tmp_path):
    # Inkling's hybrid MTP layer needs both the attention and the conv mask
    model = _source_model(tmp_path, load_mtp=True, num_mtp_layers=2)
    input_ids = torch.tensor(_calibration_dataset(num_samples=1)["input_ids"])
    num_pad = 3
    padded_ids = torch.cat([torch.full((1, num_pad), 3), input_ids], dim=1)
    attention_mask = torch.cat(
        [torch.zeros(1, num_pad, dtype=torch.long), torch.ones_like(input_ids)], dim=1
    )
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    expected = _mtp_layer_outputs(model, input_ids=input_ids)
    actual = _mtp_layer_outputs(
        model,
        input_ids=padded_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    for padded, unpadded in zip(actual, expected, strict=True):
        torch.testing.assert_close(padded[:, num_pad:], unpadded)


def test_each_mtp_layer_is_one_subgraph(tmp_path):
    model = _source_model(tmp_path, load_mtp=True, num_mtp_layers=2)
    input_ids = torch.tensor(_calibration_dataset(num_samples=1)["input_ids"])
    sample = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    with mtp_calibration_context(model):
        subgraphs = trace_subgraphs(model, sample, get_no_split_params(model), [])

    owners = {}
    for index, subgraph in enumerate(subgraphs):
        for name in subgraph.submodule_dict(model):
            owners.setdefault(name, set()).add(index)
    for layer in ("mtp.layers.0", "mtp.layers.1"):
        assert len(owners[layer]) == 1, layer
        assert owners[f"{layer}.eh_proj"] == owners[layer]
        assert owners[f"{layer}.mtp_block"] == owners[layer]
    assert owners["mtp.layers.0"] != owners["mtp.layers.1"]

    # Each subgraph must return what later subgraphs consume. The model output
    # does not depend on MTP, so it is ready before the last MTP layer runs.
    cache = IntermediatesCache.from_dataloader([sample], torch.device("cpu"))
    with torch.no_grad(), mtp_calibration_context(model):
        for subgraph in subgraphs:
            inputs = cache.fetch(0, list(subgraph.input_names))
            cache.update(0, subgraph.forward(model, **inputs))


def test_two_layer_mtp_calibrates_sequentially(tmp_path):
    model = _source_model(tmp_path, load_mtp=True, num_mtp_layers=2)
    recipe = QuantizationModifier(scheme={"NVFP4": [r"re:^mtp\.layers\..*proj$"]})
    oneshot(
        model=model,
        recipe=recipe,
        processor=_tokenizer(),
        dataset=_calibration_dataset(),
        num_calibration_samples=4,
        pipeline="sequential",
    )
    for index, layer in enumerate(model.mtp.layers):
        for name in ("eh_proj", "mtp_block.self_attn.q_proj"):
            scale = layer.get_submodule(name).input_global_scale
            assert torch.isfinite(scale).all() and (scale > 0).all(), (index, name)


def test_mtp_does_not_run_when_recipe_ignores_it(tmp_path):
    model = _source_model(tmp_path, load_mtp=True)
    calls = []
    model.mtp.layers[0].register_forward_hook(lambda *_: calls.append(1))
    recipe = QuantizationModifier(
        scheme={"NVFP4": ["Linear"]}, ignore=["lm_head", r"re:^mtp\."]
    )
    oneshot(
        model=model,
        recipe=recipe,
        processor=_tokenizer(),
        dataset=_calibration_dataset(),
        num_calibration_samples=4,
    )
    assert not calls
    assert not hasattr(model.mtp.layers[0].eh_proj, "quantization_scheme")


def test_mtp_calibration_accepts_inputs_embeds(tmp_path):
    # GLM's MTP shares the backbone embedding
    model = _loaded_mtp_model(tmp_path, "trailing")
    input_ids = torch.tensor(_calibration_dataset(num_samples=2)["input_ids"])
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)
    expected = _mtp_layer_outputs(model, input_ids=input_ids)
    actual = _mtp_layer_outputs(model, inputs_embeds=inputs_embeds)
    for embedded, tokens in zip(actual, expected, strict=True):
        torch.testing.assert_close(embedded, tokens)


def test_mtp_calibration_rejects_embeds_for_own_mtp_embedding(tmp_path):
    # Inkling's MTP normalizes embeddings with its own weights
    model = _loaded_mtp_model(tmp_path, "dedicated")
    input_ids = torch.tensor(_calibration_dataset(num_samples=1)["input_ids"])
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)
    with pytest.raises(ValueError, match="their own embedding"):
        _mtp_layer_outputs(model, inputs_embeds=inputs_embeds)


_BACKBONE = r"re:^model\.layers\.\d+\."


@pytest.mark.parametrize(
    "modifier,calibrates_mtp",
    [
        (
            SmoothQuantModifier(
                mappings=[
                    [[f"{_BACKBONE}self_attn.q_proj$"], f"{_BACKBONE}input_layernorm$"]
                ],
            ),
            False,
        ),
        (
            SmoothQuantModifier(
                mappings=[[["re:.*self_attn.q_proj$"], "re:.*input_layernorm$"]]
            ),
            True,
        ),
        (
            AWQModifier(
                mappings=[
                    AWQMapping(
                        f"{_BACKBONE}input_layernorm$",
                        [f"{_BACKBONE}self_attn.q_proj$"],
                    )
                ]
            ),
            False,
        ),
        (
            AWQModifier(
                mappings=[
                    AWQMapping("re:.*input_layernorm$", ["re:.*self_attn.q_proj$"])
                ]
            ),
            True,
        ),
        (REAPPruningModifier(sparsity=0.5, ignore=[r"^mtp\."]), False),
        (REAPPruningModifier(sparsity=0.5), True),
        (QADModifier(), False),
    ],
)
def test_calibrated_names_resolve_mtp(tmp_path, modifier, calibrates_mtp):
    model = _loaded_mtp_model(tmp_path, "trailing")
    names = _calibrated_names(model, modifier)
    # QAD only distills what a preceding modifier quantized
    assert bool(names) != isinstance(modifier, QADModifier)
    assert any(name.startswith("mtp.") for name in names) == calibrates_mtp


def test_mtp_does_not_run_for_backbone_only_mappings(tmp_path):
    model = _loaded_mtp_model(tmp_path, "trailing")
    calls = []
    model.mtp.layers[0].register_forward_hook(lambda *_: calls.append(1))
    recipe = [
        SmoothQuantModifier(
            mappings=[
                [[f"{_BACKBONE}self_attn.q_proj$"], f"{_BACKBONE}input_layernorm$"]
            ],
        ),
        QuantizationModifier(scheme="W8A8", ignore=["lm_head", r"re:^mtp\."]),
    ]
    oneshot(
        model=model,
        recipe=recipe,
        processor=_tokenizer(),
        dataset=_calibration_dataset(),
        num_calibration_samples=4,
    )
    assert not calls


@pytest.mark.parametrize("layout", ["dedicated", "trailing"])
@pytest.mark.parametrize("method", ["gptq", "autoround"])
def test_calibrated_weight_quantization_of_mtp(tmp_path, layout, method):
    from llmcompressor.modifiers.autoround import AutoRoundModifier
    from llmcompressor.modifiers.gptq import GPTQModifier

    model = _loaded_mtp_model(tmp_path, layout)
    scheme = QuantizationScheme(
        targets=[r"re:^mtp\.layers\..*proj$"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=16
        ),
    )
    if method == "gptq":
        recipe = GPTQModifier(config_groups={"group_0": scheme})
    else:
        recipe = AutoRoundModifier(
            config_groups={"group_0": scheme}, iters=2, enable_torch_compile=False
        )
    linears = [
        name
        for name, module in model.mtp.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith("proj")
    ]
    assert "layers.0.eh_proj" in linears

    oneshot(
        model=model,
        recipe=recipe,
        processor=_tokenizer(),
        dataset=_calibration_dataset(),
        num_calibration_samples=4,
    )

    for name in linears:
        module = model.mtp.get_submodule(name)
        assert module.quantization_scheme is not None, name
        assert torch.isfinite(module.weight_scale).all(), name

    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    weights = get_weight_mappings(destination)
    mtp_prefixes = ("model.mtp.", f"model.layers.{model.config.num_hidden_layers}.")
    mtp_scales = [
        name
        for name in weights
        if name.endswith(".weight_scale") and name.startswith(mtp_prefixes)
    ]
    assert len(mtp_scales) == len(linears)
    projection = "input_proj" if layout == "dedicated" else "eh_proj"
    assert any(f".{projection}.weight_scale" in name for name in mtp_scales)
    quant = json.loads((destination / "config.json").read_text())["quantization_config"]
    assert not any(
        name.removesuffix(".weight_scale") in quant["ignore"] for name in mtp_scales
    )


def test_unloaded_mtp_warns_for_default_linear_targets(tmp_path):
    model = _source_model(tmp_path)
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        oneshot(
            model=model,
            recipe=QuantizationModifier(scheme="FP8_DYNAMIC"),
            output_dir=str(tmp_path / "destination"),
        )
    finally:
        logger.remove(handler_id)
    mtp_warnings = [log for log in logs if "MTP" in log]
    assert len(mtp_warnings) == 1
    assert "Use load_context(load_mtp=True)" in mtp_warnings[0]
    assert "copied unchanged when saving" in mtp_warnings[0]


def test_copied_mtp_embedding_participates_in_offloading(tmp_path):
    source = _source_model(tmp_path)
    original = MtpModel.from_pretrained

    def load_with_copied_embedding(*args, **kwargs):
        mtp = original(*args, **kwargs)
        mtp.embed_tokens = deepcopy(mtp.embed_tokens)
        return mtp

    with (
        patch.object(
            MtpModel, "from_pretrained", side_effect=load_with_copied_embedding
        ),
        load_context(InklingForCausalLM, load_mtp=True),
    ):
        model = InklingForCausalLM.from_pretrained(source.name_or_path)
    assert model.mtp.embed_tokens is not model.get_input_embeddings()
    assert isinstance(model.mtp.embed_tokens._parameters, OffloadCache)
    assert "mtp.embed_tokens.weight" in model.state_dict()

    destination = tmp_path / "destination"
    model.save_pretrained(destination)
    assert "model.mtp.embed_tokens.weight" not in get_weight_mappings(destination)
    assert isinstance(model.mtp.embed_tokens._parameters, OffloadCache)


@pytest.mark.parametrize("model_cls", [InklingForCausalLM, Glm4MoeForCausalLM])
def test_load_context_supports_normal_save_before_oneshot(tmp_path, model_cls):
    source = (
        _source_model(tmp_path)
        if model_cls is InklingForCausalLM
        else _source_glm_model(tmp_path, model_cls)
    )
    with (
        patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
        patch("transformers.utils.hub.is_offline_mode", return_value=True),
        load_context(model_cls),
    ):
        model = model_cls.from_pretrained(source.name_or_path, local_files_only=True)
    destination = tmp_path / "destination"
    model.save_pretrained(destination, max_shard_size="100KB")
    expected = load_file(tmp_path / "source" / "model.safetensors")
    saved = get_weight_mappings(destination)
    assert saved.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(load_file(saved[name])[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("load_mtp", [False, True])
def test_mtp_disk_offload_matches_cpu_save(tmp_path, load_mtp):
    source = _source_glm_model(tmp_path)
    saved = []
    for device_map in ("cpu", "auto_offload"):
        reset_session()
        kwargs = (
            {"max_memory": {"cpu": "2MB"}, "offload_folder": str(tmp_path / "offload")}
            if device_map == "auto_offload"
            else {}
        )
        with load_context(Glm4MoeForCausalLM, load_mtp=load_mtp):
            model = Glm4MoeForCausalLM.from_pretrained(
                source.name_or_path,
                device_map=device_map,
                local_files_only=True,
                **kwargs,
            )
        assert hasattr(model, "mtp") == load_mtp
        if device_map == "auto_offload":
            assert any(
                name.startswith("model.layers.") and offload == "disk"
                for name, (_, offload) in get_device_map(model).items()
            )
        oneshot(
            model=model,
            recipe=QuantizationModifier(
                scheme="FP8_DYNAMIC", ignore=["lm_head", r"re:.*\.eh_proj$"]
            ),
        )
        destination = tmp_path / device_map
        model.save_pretrained(destination, max_shard_size="1MB")
        mapping = get_weight_mappings(destination)
        saved.append(
            {
                name: tensor
                for shard in set(mapping.values())
                for name, tensor in load_file(shard).items()
            }
        )

    cpu, disk = saved
    assert cpu.keys() == disk.keys()
    mtp_prefix = f"model.layers.{source.config.num_hidden_layers}."
    assert any(name.startswith(mtp_prefix) for name in disk)
    for name in cpu:
        torch.testing.assert_close(
            cpu[name].float(), disk[name].float(), rtol=0, atol=0
        )


def _source_glm_model(tmp_path, model_cls=Glm4MoeForCausalLM, first_k_dense_replace=0):
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
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=first_k_dense_replace,
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
    # Build a real MoE checkpoint even when older MtpModel creates a dense block.
    mtp.layers[0].mtp_block = deepcopy(model.get_decoder().layers[-1])
    # MtpModel's generic initializer does not initialize architecture-specific
    # fused expert tensors, which are allocated with torch.empty.
    mtp.layers.apply(model._init_weights)
    linearize_moe(mtp)
    # Keep the fixture's checkpoint layout independent of mappings registered
    # by earlier load_context calls in this process.
    linearize_moe(model)
    source = tmp_path / "source"
    model.save_pretrained(source)
    weights = load_file(source / "model.safetensors")
    assert "mtp_block.mlp.experts.0.up_proj.weight" in mtp.layers[0].state_dict()
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
    model = _source_glm_model(tmp_path, model_cls, first_k_dense_replace=1)
    source = model.name_or_path
    prefix = f"model.layers.{model.config.num_hidden_layers}."
    mtp_config = model.config.get_mtp_config()
    supported = (
        getattr(mtp_config, "mlp_layer_types", None) is not None
        or getattr(mtp_config, "first_k_dense_replace", 0) == 0
    )
    with load_context(model_cls, load_mtp=target_mtp):
        model = model_cls.from_pretrained(source)
    assert hasattr(model, "mtp") == (target_mtp and supported)

    recipe = QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)

    destination = tmp_path / "destination"
    model.save_pretrained(destination, max_shard_size="100KB")
    weights = get_weight_mappings(destination)
    assert len(set(weights.values())) > 1
    if target_mtp and supported:
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
        expected = {
            name: tensor
            for name, tensor in load_file(
                tmp_path / "source" / "model.safetensors"
            ).items()
            if name.startswith(prefix)
        }
        torch.testing.assert_close(
            load_file(destination / "model_mtp.safetensors"), expected, rtol=0, atol=0
        )
        quant = json.loads((destination / "config.json").read_text())[
            "quantization_config"
        ]
        assert all(
            name.removesuffix(".weight") in quant["ignore"]
            for name in expected
            if name.endswith(".weight")
        )
    assert not any(name.startswith("mtp.") for name in weights)
