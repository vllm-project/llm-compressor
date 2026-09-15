import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForCausalLM,
    FineGrainedFP8Config,
    PreTrainedTokenizerFast,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)

from llmcompressor import oneshot
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.entrypoints.utils import post_process
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _remove_fp8_save_roundtrip,
)
from llmcompressor.utils import load_context


def test_post_process_uses_wrapped_save_once(tmp_path):
    """The save wrapper owns MTP for both oneshot and deferred saves."""
    model = Mock()
    model_args = SimpleNamespace(
        model=model,
        processor=None,
        save_compressed=True,
        model_revision="revision",
    )
    post_process(
        model_args=model_args,
        output_dir=str(tmp_path),
    )

    model.save_pretrained.assert_called_once_with(str(tmp_path), save_compressed=True)


@pytest.mark.parametrize(
    "residual", ["is_quantized", "hf_quantizer", "config", "weight", "buffer"]
)
def test_native_fp8_guard_rejects_residual_quantization(tmp_path, residual):
    config = Qwen3_5TextConfig()
    config.quantization_config = {"quant_method": "fp8"}
    config.save_pretrained(tmp_path)
    model = torch.nn.Linear(32, 32)
    model.config = Qwen3_5TextConfig(name_or_path=str(tmp_path))
    model.is_quantized = False
    if residual == "is_quantized":
        model.is_quantized = True
    elif residual == "hf_quantizer":
        model.hf_quantizer = object()
    elif residual == "config":
        model.config.quantization_config = {"quant_method": "fp8"}
    elif residual == "buffer":
        model.register_buffer("weight_scale_inv", torch.ones(1, 1))
    else:
        model.weight = torch.nn.Parameter(
            model.weight.to(torch.float8_e4m3fn), requires_grad=False
        )
    owner = SimpleNamespace(model_args=SimpleNamespace(trust_remote_code_model=False))
    with pytest.raises(ValueError, match="different format"):
        Oneshot.validate_model(owner, model)


def test_fp8_save_cleanup_retains_scoped_reshapes_and_renames():
    from transformers.core_model_loading import (
        Concatenate,
        WeightConverter,
        WeightRenaming,
    )
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    dequantize = Fp8Dequantize(None)
    concatenate = Concatenate(dim=0)
    rename = WeightRenaming("old_prefix", "new_prefix")
    fused = WeightConverter(
        ["q.weight", "q.weight_scale_inv", "k.weight", "k.weight_scale_inv"],
        "qk.weight",
        [dequantize, concatenate],
    )
    fused.scope_prefix, fused.base_model_prefix = "language_model", "model"
    model = SimpleNamespace(_weight_conversions=[rename, fused])
    _remove_fp8_save_roundtrip(model)
    assert model._weight_conversions[0] is rename
    updated = model._weight_conversions[1]
    assert updated.operations == [concatenate]
    assert updated._original_source_patterns == ["q.weight", "k.weight"]
    assert updated.scope_prefix == "language_model"
    assert updated.base_model_prefix == "model"
    assert fused.operations == [dequantize, concatenate]


@pytest.mark.parametrize(
    "deferred_save", [False, True], ids=["output-dir", "save-pretrained"]
)
@pytest.mark.parametrize(
    "moe,scheme,native_fp8,quantize_backbone",
    [
        (False, None, False, True),
        (False, "NVFP4A16", False, True),
        (True, None, False, True),
        (False, "BF16", False, True),
        (False, None, True, True),
        (False, None, True, False),
        (False, "BF16", True, True),
        (False, "BF16", True, False),
        (False, "NVFP4A16", True, True),
        (False, "NVFP4A16", True, False),
    ],
)
def test_qwen_real_load_and_save(
    tmp_path, deferred_save, moe, scheme, native_fp8, quantize_backbone
):
    """Exercise the instantiated CausalLM aliases, not a hand-set loaded config."""
    config_class = Qwen3_5MoeTextConfig if moe else Qwen3_5TextConfig
    extra = (
        dict(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
        )
        if moe
        else dict(intermediate_size=64)
    )
    config = config_class(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["full_attention"],
        **extra,
    )
    config.mtp_num_hidden_layers = 1
    source, destination = tmp_path / "source", tmp_path / "destination"
    AutoModelForCausalLM.from_config(config).save_pretrained(source)
    # The checkpoint advertises the multimodal class, but Transformers loads the
    # text class. Its real save rewrites architectures to that instantiated class.
    metadata = json.loads((source / "config.json").read_text())
    metadata["architectures"] = None
    metadata = {
        "architectures": [f"Qwen3_5{'Moe' if moe else ''}ForConditionalGeneration"],
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": metadata,
    }
    (source / "config.json").write_text(json.dumps(metadata))
    tensors = {
        **{
            f"mtp.layers.0.self_attn.{proj}_proj.weight": torch.randn(32, 32)
            for proj in ("q", "k", "v")
        },
        "mtp.fc.weight": torch.randn(32, 64),
        "mtp.norm.weight": torch.randn(32),
    }
    weights = load_file(source / "model.safetensors")
    weights.update(tensors)
    load_kwargs = {}
    if native_fp8:
        # Store a real native-FP8 backbone and MTP, then use Transformers'
        # explicit dequantizing load. No validation or saving hooks are mocked.
        modules = []
        for name, weight in list(weights.items()):
            if ".self_attn." in name and name.endswith(".weight") and weight.ndim == 2:
                modules.append(name.removesuffix(".weight"))
                weights[name] = weight.to(torch.float8_e4m3fn)
                weights[name.removesuffix(".weight") + ".weight_scale_inv"] = (
                    torch.ones(
                        (weight.shape[0] + 31) // 32, (weight.shape[1] + 31) // 32
                    )
                )
        metadata["quantization_config"] = {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "modules_to_not_convert": [
                name.removesuffix(".weight")
                for name, value in weights.items()
                if name.endswith(".weight")
                and value.ndim == 2
                and name.removesuffix(".weight") not in modules
            ],
        }
        (source / "config.json").write_text(json.dumps(metadata))
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    save_file(weights, source / "model.safetensors")
    with load_context(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, **load_kwargs
        )
    assert type(model).__name__ == f"Qwen3_5{'Moe' if moe else ''}ForCausalLM"
    assert not any(name.startswith("mtp.") for name, _ in model.named_parameters())
    processor = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    oneshot(
        model=model,
        processor=processor,
        recipe=QuantizationModifier(
            targets=["re:.*self_attn.*_proj$"], scheme="FP8_DYNAMIC"
        )
        if quantize_backbone
        else [],
        mtp_scheme=scheme,
        output_dir=None if deferred_save else str(destination),
    )
    if deferred_save:
        model.save_pretrained(destination)
    if native_fp8:
        backbone = load_file(destination / "model.safetensors")
        assert not any(name.endswith("weight_scale_inv") for name in backbone)
        embedding = next(
            value
            for name, value in backbone.items()
            if name.endswith("embed_tokens.weight")
        )
        original = next(
            value
            for name, value in weights.items()
            if name.endswith("embed_tokens.weight")
        )
        assert torch.equal(embedding, original.bfloat16())
    saved = load_file(destination / "model_mtp.safetensors")
    quantization = json.loads((destination / "config.json").read_text()).get(
        "quantization_config", {}
    )
    group = quantization.get("config_groups", {}).get("mtp_group")
    if native_fp8 and scheme is None:
        assert group["format"] == "float-quantized"
        assert group["weights"]["block_structure"] == [32, 32]
        for name, value in tensors.items():
            if ".self_attn." in name:
                assert saved[name].dtype == torch.float8_e4m3fn
                assert torch.equal(saved[name].float(), weights[name].float())
                assert torch.equal(
                    saved[name.removesuffix(".weight") + ".weight_scale"],
                    weights[name.removesuffix(".weight") + ".weight_scale_inv"],
                )
            else:
                assert torch.equal(saved[name], value)
    elif scheme == "NVFP4A16":
        assert saved["mtp.layers.0.self_attn.q_proj.weight_packed"].dtype == torch.uint8
        assert group["format"] == "nvfp4-pack-quantized"
    else:
        assert group is None
        assert saved.keys() == tensors.keys()
        for name in tensors:
            if native_fp8 and ".self_attn." in name:
                assert saved[name].dtype == torch.bfloat16
                assert torch.equal(saved[name], weights[name].bfloat16())
            else:
                expected = (
                    tensors[name].bfloat16() if scheme == "BF16" else tensors[name]
                )
                assert torch.equal(saved[name], expected)
