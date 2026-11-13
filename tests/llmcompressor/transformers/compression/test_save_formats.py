import json

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from safetensors import safe_open
from transformers import AutoModelForCausalLM, Qwen3Config

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

WEIGHT = "model.layers.0.self_attn.k_proj.weight"


def _make_model(tmp_path, explicit_format=None):
    torch.manual_seed(417)
    torch.set_num_threads(2)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        dtype="float32",
    )
    source = tmp_path / "source"
    config.save_pretrained(source)
    config.name_or_path = str(source)
    model = AutoModelForCausalLM.from_config(config).eval()
    scheme = preset_name_to_scheme("W8A8", ["Linear"])
    scheme.format = explicit_format
    oneshot(
        model=model,
        recipe=QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=["lm_head"],
        ),
    )
    return model


def _read_checkpoint(directory):
    config = json.loads((directory / "config.json").read_text())["quantization_config"]
    with safe_open(directory / "model.safetensors", framework="pt") as tensors:
        weight = tensors.get_tensor(WEIGHT)
    return config, weight


@pytest.mark.parametrize("explicit_format", [None, "int-quantized", "dense"])
def test_frozen_save_does_not_change_later_compressed_format(tmp_path, explicit_format):
    model = _make_model(tmp_path, explicit_format)
    scheme = model.model.layers[0].self_attn.k_proj.quantization_scheme
    original_format = scheme.format

    model.save_pretrained(tmp_path / "frozen", save_compressed=False)
    assert scheme.format == original_format

    frozen_config, frozen_weight = _read_checkpoint(tmp_path / "frozen")
    assert frozen_config["format"] == "dense"
    assert frozen_config["quantization_status"] == "frozen"
    assert frozen_weight.dtype == torch.float32

    model.save_pretrained(tmp_path / "compressed", save_compressed=True)
    config, weight = _read_checkpoint(tmp_path / "compressed")
    expected_format = explicit_format or "int-quantized"
    expected_dtype = torch.float32 if expected_format == "dense" else torch.int8
    assert config["format"] == expected_format
    assert config["quantization_status"] == "compressed"
    assert weight.dtype == expected_dtype


def test_frozen_save_restores_scheme_formats_after_failure(tmp_path):
    model = _make_model(tmp_path)
    schemes = [
        module.quantization_scheme
        for module in model.modules()
        if getattr(module, "quantization_scheme", None) is not None
    ]
    formats = [scheme.format for scheme in schemes]

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("A file cannot be a checkpoint directory.\n")

    with pytest.raises((OSError, ValueError)):
        model.save_pretrained(blocked, save_compressed=False)

    assert [scheme.format for scheme in schemes] == formats
