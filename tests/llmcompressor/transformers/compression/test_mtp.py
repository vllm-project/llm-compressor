import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils.match import match_name
from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.transformers.compression.mtp import save_mtp_tensors


def _scheme(name="FP8_DYNAMIC", targets=None):
    scheme = preset_name_to_scheme(name, targets or ["Linear"])
    scheme.format = infer_module_format(torch.nn.Linear, scheme).value
    return scheme


def _quant_config(group="backbone", scheme=None):
    scheme = scheme or _scheme()
    return QuantizationConfig(
        config_groups={group: scheme},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=scheme.format,
        ignore=[],
    )


def _checkpoint(path, config, tensors):
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    with open(path / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file)


def _destination(path):
    config = {"quantization_config": _quant_config().model_dump(mode="json")}
    _checkpoint(path, config, {"model.layers.0.weight": torch.ones(2, 2)})


def _model(source):
    return SimpleNamespace(
        name_or_path=str(source),
        config=SimpleNamespace(_name_or_path=str(source)),
    )


def _saved_tensors(destination):
    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        return {name: file.get_tensor(name) for name in file.keys()}


def _saved_config(destination):
    with open(destination / "config.json", encoding="utf-8") as file:
        return json.load(file)["quantization_config"]


def test_none_copies_mtp_exactly(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    tensors = {
        "model.layers.0.weight": torch.zeros(2, 2),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(4, 4),
        "mtp.fc.weight": torch.randn(4, 4),
    }
    _checkpoint(source, {"mtp_num_hidden_layers": 1}, tensors)
    _destination(destination)

    save_mtp_tensors(_model(source), destination)

    saved = _saved_tensors(destination)
    assert saved.keys() == tensors.keys() - {"model.layers.0.weight"}
    for name, tensor in saved.items():
        assert torch.equal(tensor, tensors[name])
        assert tensor.dtype == tensors[name].dtype

    config = _saved_config(destination)
    assert config["ignore"] == [r"re:^mtp(?:\.|$)"]
    with open(destination / "model.safetensors.index.json") as file:
        weight_map = json.load(file)["weight_map"]
    assert weight_map["mtp.fc.weight"] == "model_mtp.safetensors"
    assert weight_map["model.layers.0.weight"] == "model.safetensors"


def test_none_preserves_source_mtp_scheme(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    targets = ["mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate"]
    scheme = _scheme(targets=targets)
    quant_config = _quant_config("mtp_group", scheme)
    tensors = ModelFreePtqConverter(quant_config).process(
        {f"{target}.weight": torch.randn(4, 4) for target in targets}
    )
    _checkpoint(
        source,
        {
            "mtp_num_hidden_layers": 1,
            "quantization_config": quant_config.model_dump(mode="json"),
        },
        tensors,
    )
    _destination(destination)

    save_mtp_tensors(_model(source), destination)

    saved = _saved_tensors(destination)
    assert saved.keys() == tensors.keys()
    assert all(torch.equal(saved[name], tensor) for name, tensor in tensors.items())
    assert (
        _saved_config(destination)["config_groups"]["mtp_group"]["targets"] == targets
    )


def test_bf16_dequantizes_native_fp8(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    config = {
        "mtp_num_hidden_layers": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [2, 2],
        },
    }
    tensors = {
        "mtp.layers.0.mlp.down_proj.weight": torch.ones(
            4, 4, dtype=torch.float8_e4m3fn
        ),
        "mtp.layers.0.mlp.down_proj.weight_scale_inv": torch.full((2, 2), 2.0),
    }
    _checkpoint(source, config, tensors)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, "bf16")

    saved = _saved_tensors(destination)
    assert set(saved) == {"mtp.layers.0.mlp.down_proj.weight"}
    assert saved["mtp.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16
    assert torch.equal(
        saved["mtp.layers.0.mlp.down_proj.weight"],
        torch.full((4, 4), 2.0, dtype=torch.bfloat16),
    )


def test_bf16_dequantizes_compressed_tensors_fp8(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    scheme = _scheme(targets=[r"re:^mtp\."])
    quant_config = _quant_config("mtp_group", scheme)
    converter = ModelFreePtqConverter(quant_config)
    original = torch.randn(4, 4, dtype=torch.bfloat16)
    tensors = converter.process({"mtp.layers.0.mlp.down_proj.weight": original})
    config = {
        "mtp_num_hidden_layers": 1,
        "quantization_config": quant_config.model_dump(mode="json"),
    }
    _checkpoint(source, config, tensors)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, "bf16")

    saved = _saved_tensors(destination)
    assert set(saved) == {"mtp.layers.0.mlp.down_proj.weight"}
    assert saved["mtp.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16
    assert torch.allclose(
        saved["mtp.layers.0.mlp.down_proj.weight"], original, atol=0.25
    )


@pytest.mark.parametrize(
    ("scheme", "compressed_param"),
    [
        ("FP8_DYNAMIC", "weight_scale"),
        ("FP8_BLOCK", "weight_scale"),
        ("NVFP4A16", "weight_packed"),
        ("MXFP4", "weight_packed"),
    ],
)
def test_quantizes_mtp_with_model_free_converter(tmp_path, scheme, compressed_param):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    tensors = {
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(128, 128),
        "mtp.fc.weight": torch.randn(128, 128),
        "mtp.pre_fc_norm_embedding.weight": torch.randn(128),
    }
    _checkpoint(source, {"mtp_num_hidden_layers": 1}, tensors)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, scheme)

    saved = _saved_tensors(destination)
    assert f"mtp.layers.0.mlp.down_proj.{compressed_param}" in saved
    assert saved["mtp.fc.weight"].dtype == torch.bfloat16
    assert torch.equal(
        saved["mtp.pre_fc_norm_embedding.weight"],
        tensors["mtp.pre_fc_norm_embedding.weight"].to(torch.bfloat16),
    )

    config = _saved_config(destination)
    assert list(config["config_groups"])[0] == "mtp_group"
    assert config["config_groups"]["mtp_group"]["targets"] == [r"re:^mtp\."]
    assert any(match_name("mtp.fc", target) for target in config["ignore"])
    assert "mtp.pre_fc_norm_embedding" in config["ignore"]


def test_nvfp4_fuses_dsa_scales(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    prefix = "model.layers.2.mlp"
    gate = torch.full((2, 32, 32), 0.1)
    up = torch.full((2, 32, 32), 10.0)
    tensors = {
        f"{prefix}.experts.gate_up_proj.weight": torch.cat((gate, up), dim=1),
        f"{prefix}.experts.down_proj.weight": torch.ones(2, 32, 32),
        f"{prefix}.shared_experts.gate_proj.weight": torch.full((32, 32), 0.1),
        f"{prefix}.shared_experts.up_proj.weight": torch.full((32, 32), 10.0),
        "model.layers.2.self_attn.q_a_proj.weight": torch.full((32, 32), 0.1),
        "model.layers.2.self_attn.kv_a_proj_with_mqa.weight": torch.full(
            (32, 32), 10.0
        ),
    }
    _checkpoint(
        source,
        {"num_hidden_layers": 2, "num_nextn_predict_layers": 1},
        tensors,
    )
    _destination(destination)

    save_mtp_tensors(_model(source), destination, "NVFP4A16")

    saved = _saved_tensors(destination)
    for expert in ("experts.0", "experts.1", "shared_experts"):
        gate_scale = saved[f"{prefix}.{expert}.gate_proj.weight_global_scale"]
        up_scale = saved[f"{prefix}.{expert}.up_proj.weight_global_scale"]
        assert torch.equal(gate_scale, up_scale)
    attn = "model.layers.2.self_attn"
    assert torch.equal(
        saved[f"{attn}.q_a_proj.weight_global_scale"],
        saved[f"{attn}.kv_a_proj_with_mqa.weight_global_scale"],
    )


def test_extra_language_model_layer_uses_runtime_names(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    prefix = "model.language_model.layers.2"
    tensors = {
        f"{prefix}.mlp.experts.0.down_proj.weight": torch.randn(4, 4),
        f"{prefix}.self_attn.q_proj.weight": torch.randn(4, 4),
        f"{prefix}.eh_proj.weight": torch.randn(4, 4),
    }
    _checkpoint(
        source,
        {"num_hidden_layers": 2, "num_nextn_predict_layers": 1},
        tensors,
    )
    _checkpoint(
        destination,
        {
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "quantization_config": _quant_config().model_dump(mode="json"),
        },
        {"model.layers.0.weight": torch.ones(2, 2)},
    )

    save_mtp_tensors(_model(source), destination, "FP8_DYNAMIC")

    saved = _saved_tensors(destination)
    assert f"{prefix}.mlp.experts.0.down_proj.weight_scale" in saved
    assert saved[f"{prefix}.self_attn.q_proj.weight"].dtype == torch.bfloat16
    config = _saved_config(destination)
    assert config["config_groups"]["mtp_group"]["targets"] == [
        r"re:^model\.layers\.2\."
    ]
    assert any(
        match_name("model.layers.2.self_attn.q_proj", target)
        for target in config["ignore"]
    )
    assert any(
        match_name("model.layers.2.eh_proj", target) for target in config["ignore"]
    )

    copied = tmp_path / "copied"
    _destination(copied)
    save_mtp_tensors(_model(destination), copied)
    copied_config = _saved_config(copied)
    assert any(
        match_name("model.layers.2.self_attn.q_proj", target)
        for target in copied_config["ignore"]
    )


@pytest.mark.parametrize("value", ["not-a-scheme", "FP8", 8])
def test_rejects_invalid_scheme(tmp_path, value):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _checkpoint(
        source,
        {"mtp_num_hidden_layers": 1},
        {"mtp.layers.0.mlp.down_proj.weight": torch.randn(4, 4)},
    )
    _destination(destination)

    with pytest.raises((TypeError, ValueError)):
        save_mtp_tensors(_model(source), destination, value)
