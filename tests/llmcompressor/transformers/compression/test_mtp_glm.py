import json
from types import SimpleNamespace

import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.transformers.compression.mtp import save_mtp_tensors

PREFIX = "model.layers.2"


def _scheme():
    scheme = preset_name_to_scheme("FP8_DYNAMIC", targets=["Linear"])
    scheme.format = infer_module_format(torch.nn.Linear, scheme).value
    return scheme


def _quant_config():
    scheme = _scheme()
    return QuantizationConfig(
        config_groups={"backbone": scheme},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=scheme.format,
        ignore=[],
    )


def _checkpoint(path, tensors, quant=None):
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    config = {}
    if quant is not None:
        config["quantization_config"] = quant.model_dump(mode="json")
    with open(path / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file)


def _model(source):
    text_config = SimpleNamespace(num_hidden_layers=2, num_mtp_layers=1)
    return SimpleNamespace(
        name_or_path=str(source),
        config=SimpleNamespace(get_text_config=lambda: text_config),
    )


def _destination(path):
    _checkpoint(path, {"model.layers.0.weight": torch.ones(2, 2)}, _quant_config())


def _saved(path):
    with safe_open(path / "model_mtp.safetensors", framework="pt") as shard:
        return {name: shard.get_tensor(name) for name in shard.keys()}


def _config(path):
    with open(path / "config.json", encoding="utf-8") as file:
        return json.load(file)["quantization_config"]


def test_none_copies_glm_mtp_exactly(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    tensors = {
        "model.layers.0.weight": torch.zeros(2, 2),
        f"{PREFIX}.self_attn.q_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        f"{PREFIX}.eh_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    _checkpoint(source, tensors)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, None)

    saved = _saved(destination)
    assert saved.keys() == tensors.keys() - {"model.layers.0.weight"}
    assert all(torch.equal(saved[name], tensors[name]) for name in saved)
    assert r"re:^model\.layers\.2\." in _config(destination)["ignore"]
    mappings = get_weight_mappings(destination)
    assert f"{PREFIX}.eh_proj.weight" in mappings
    assert "model.layers.0.weight" in mappings


def test_none_preserves_compressed_mtp_scheme(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    scheme = _scheme()
    source_quant = QuantizationConfig(
        config_groups={"mtp_group": scheme},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=scheme.format,
        ignore=[],
    )
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    tensors = ModelFreePtqConverter(source_quant).process(
        {f"{PREFIX}.self_attn.q_proj.weight": weight}
    )
    _checkpoint(source, tensors, source_quant)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, None)

    saved = _saved(destination)
    assert saved.keys() == tensors.keys()
    assert all(torch.equal(saved[name], tensors[name]) for name in saved)
    config = _config(destination)
    assert next(iter(config["config_groups"])) == "mtp_group"
    assert config["config_groups"]["mtp_group"]["targets"] == [
        r"re:^model\.layers\.2\."
    ]


def test_bf16_dequantizes_compressed_mtp(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    scheme = _scheme()
    source_quant = QuantizationConfig(
        config_groups={"mtp_group": scheme},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=scheme.format,
        ignore=[],
    )
    tensors = ModelFreePtqConverter(source_quant).process(
        {f"{PREFIX}.self_attn.q_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16)}
    )
    _checkpoint(source, tensors, source_quant)
    _destination(destination)

    save_mtp_tensors(_model(source), destination, "bf16")

    saved = _saved(destination)
    assert set(saved) == {f"{PREFIX}.self_attn.q_proj.weight"}
    assert saved[f"{PREFIX}.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert torch.isfinite(saved[f"{PREFIX}.self_attn.q_proj.weight"]).all()
    assert r"re:^model\.layers\.2\." in _config(destination)["ignore"]


def test_bf16_dequantizes_native_fp8_block(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _checkpoint(
        source,
        {
            f"{PREFIX}.self_attn.q_proj.weight": torch.ones(
                4, 4, dtype=torch.float8_e4m3fn
            ),
            f"{PREFIX}.self_attn.q_proj.weight_scale_inv": torch.full((2, 2), 2.0),
        },
    )
    with open(source / "config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [2, 2],
                }
            },
            file,
        )
    _destination(destination)

    save_mtp_tensors(_model(source), destination, "bf16")

    saved = _saved(destination)
    assert set(saved) == {f"{PREFIX}.self_attn.q_proj.weight"}
    assert torch.equal(
        saved[f"{PREFIX}.self_attn.q_proj.weight"],
        torch.full((4, 4), 2.0, dtype=torch.bfloat16),
    )
