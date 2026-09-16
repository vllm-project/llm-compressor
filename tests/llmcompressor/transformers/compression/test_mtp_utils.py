import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    preset_name_to_scheme,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

from llmcompressor.transformers.compression import mtp
from llmcompressor.transformers.compression.mtp import (
    _dequantize_fp8_blocks,
    _partition_mtp_tensors,
    _quantize_and_save_mtp_tensors,
    _resolve_mtp_layout,
    _resolve_mtp_scheme,
    save_mtp_tensors,
)


def _qwen_case() -> tuple[PretrainedConfig, dict[str, torch.Tensor], str, str]:
    config = PretrainedConfig(
        architectures=["Qwen3_5ForConditionalGeneration"],
        mtp_num_hidden_layers=1,
    )
    prefix = "mtp.layers.0"
    tensors = {
        "mtp.fc.weight": torch.randn(32, 64),
        "mtp.norm.weight": torch.randn(32),
        **{
            f"{prefix}.self_attn.{proj}.weight": torch.randn(32, 32)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        f"{prefix}.mlp.gate_proj.weight": torch.randn(64, 32),
        f"{prefix}.mlp.up_proj.weight": torch.randn(64, 32),
        f"{prefix}.mlp.down_proj.weight": torch.randn(32, 64),
    }
    return (
        config,
        tensors,
        f"{prefix}.self_attn.q_proj",
        "mtp.fc",
    )


def _glm5_next_case() -> tuple[PretrainedConfig, dict[str, torch.Tensor], str, str]:
    config = PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        num_hidden_layers=5,
        num_nextn_predict_layers=1,
    )
    prefix = "model.language_model.layers.5"
    tensors = {
        f"{prefix}.eh_proj.weight": torch.randn(32, 64),
        f"{prefix}.enorm.weight": torch.randn(32),
        f"{prefix}.hc_attn_base": torch.randn(3),
        f"{prefix}.hc_ffn_fn": torch.randn(3, 32),
        f"{prefix}.mlp.gate.weight": torch.randn(2, 32),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(32, 32),
        **{
            f"{prefix}.self_attn.{proj}.weight": torch.randn(32, 32)
            for proj in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "o_proj",
            )
        },
        **{
            f"{prefix}.mlp.experts.0.{proj}.weight": torch.randn(32, 32)
            for proj in ("gate_proj", "up_proj", "down_proj")
        },
        **{
            f"{prefix}.mlp.shared_experts.{proj}.weight": torch.randn(32, 32)
            for proj in ("gate_proj", "up_proj", "down_proj")
        },
    }
    return (
        config,
        tensors,
        f"{prefix}.mlp.experts.0.gate_proj",
        f"{prefix}.self_attn.q_a_proj",
    )


def _glm_moe_dsa_case() -> tuple[PretrainedConfig, dict[str, torch.Tensor], str, str]:
    config = PretrainedConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=5,
        num_nextn_predict_layers=1,
    )
    prefix = "model.layers.5"
    tensors = {
        f"{prefix}.eh_proj.weight": torch.randn(32, 64),
        f"{prefix}.input_layernorm.weight": torch.randn(32),
        f"{prefix}.mlp.gate.weight": torch.randn(2, 32),
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(32, 32),
        **{
            f"{prefix}.self_attn.{proj}.weight": torch.randn(32, 32)
            for proj in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
        },
        **{
            f"{prefix}.mlp.experts.0.{proj}.weight": torch.randn(32, 32)
            for proj in ("gate_proj", "up_proj", "down_proj")
        },
        **{
            f"{prefix}.mlp.shared_experts.{proj}.weight": torch.randn(32, 32)
            for proj in ("gate_proj", "up_proj", "down_proj")
        },
    }
    return (
        config,
        tensors,
        f"{prefix}.self_attn.q_a_proj",
        f"{prefix}.eh_proj",
    )


def _nemotron_case() -> tuple[PretrainedConfig, dict[str, torch.Tensor], str, str]:
    config = PretrainedConfig(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
    )
    tensors = {
        "mtp.layers.0.eh_proj.weight": torch.randn(32, 64),
        "mtp.layers.0.enorm.weight": torch.randn(32),
        "mtp.layers.0.hnorm.weight": torch.randn(32),
        "mtp.layers.1.final_layernorm.weight": torch.randn(32),
        "mtp.layers.1.mixer.gate.weight": torch.randn(2, 32),
        **{
            f"mtp.layers.0.mixer.{proj}.weight": torch.randn(32, 32)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        **{
            f"mtp.layers.1.mixer.experts.0.{proj}.weight": torch.randn(32, 32)
            for proj in ("up_proj", "down_proj")
        },
        **{
            f"mtp.layers.1.mixer.shared_experts.{proj}.weight": torch.randn(32, 32)
            for proj in ("up_proj", "down_proj")
        },
    }
    return (
        config,
        tensors,
        "mtp.layers.0.eh_proj",
        "mtp.layers.1.mixer.gate",
    )


CASES: dict[
    str, Callable[[], tuple[PretrainedConfig, dict[str, torch.Tensor], str, str]]
] = {
    "qwen3.5": _qwen_case,
    "glm5-next": _glm5_next_case,
    "glm-moe-dsa": _glm_moe_dsa_case,
    "nemotron-h": _nemotron_case,
}


def _write_source(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    save_file({"backbone.weight": torch.randn(32, 32)}, path / "model.safetensors")
    save_file(tensors, path / "model_mtp.safetensors")
    weight_map = {"backbone.weight": "model.safetensors"}
    weight_map.update({name: "model_mtp.safetensors" for name in tensors})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _write_destination(path: Path, stale_mtp_name: str) -> None:
    path.mkdir()
    save_file({"backbone.weight": torch.randn(32, 32)}, path / "model.safetensors")
    scheme = preset_name_to_scheme("FP8_DYNAMIC", targets=["Linear"])
    quantization_config = QuantizationConfig(
        config_groups={"group_0": scheme},
        format="float-quantized",
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=["lm_head"],
    ).model_dump(mode="json", exclude_none=True)
    quantization_config.update(
        {
            "compressed-tensors_version": "test-version",
            "transform_config": {},
        }
    )
    (path / "config.json").write_text(
        json.dumps({"quantization_config": quantization_config})
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "backbone.weight": "model.safetensors",
                    stale_mtp_name: "stale-mtp.safetensors",
                },
            }
        )
    )


@pytest.mark.parametrize("case_name", CASES)
def test_nvfp4a16_quantizes_supported_architecture_layouts(tmp_path, case_name):
    """Each supported checkpoint layout is packed and described for its runtime."""
    config, tensors, quantized_module, dense_module = CASES[case_name]()
    layout = _resolve_mtp_layout(config, set(tensors))
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    stale_name = f"{layout.source_prefixes[0]}.obsolete.weight"
    _write_source(source, tensors)
    _write_destination(destination, stale_name)

    _quantize_and_save_mtp_tensors(
        str(source),
        str(destination),
        config,
        mtp_quant_scheme="NVFP4A16",
    )

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        output_names = set(file.keys())
    assert f"{quantized_module}.weight_packed" in output_names
    assert f"{quantized_module}.weight_global_scale" in output_names
    assert f"{dense_module}.weight" in output_names
    assert f"{dense_module}.weight_packed" not in output_names
    assert not any(name.endswith("input_global_scale") for name in output_names)

    output_config = json.loads((destination / "config.json").read_text())
    quantization_config = output_config["quantization_config"]
    assert next(iter(quantization_config["config_groups"])) == "mtp_group"
    assert quantization_config["config_groups"]["mtp_group"]["targets"] == list(
        layout.targets
    )
    assert quantization_config["compressed-tensors_version"] == "test-version"
    assert quantization_config["format"] == "mixed-precision"
    assert "lm_head" in quantization_config["ignore"]

    output_index = json.loads(
        (destination / "model.safetensors.index.json").read_text()
    )["weight_map"]
    assert stale_name not in output_index
    assert all(output_index[name] == "model_mtp.safetensors" for name in output_names)


@pytest.mark.parametrize(
    "scheme,weight_suffix",
    [
        ("FP8_DYNAMIC", ".weight"),
        ("FP8_BLOCK", ".weight"),
        ("MXFP4", ".weight_packed"),
    ],
)
@pytest.mark.parametrize("case_name", CASES)
def test_data_free_mtp_schemes(tmp_path, scheme, weight_suffix, case_name):
    """Supported data-free schemes produce an MTP checkpoint."""
    config, tensors, quantized_module, _ = CASES[case_name]()
    layout = _resolve_mtp_layout(config, set(tensors))
    if scheme == "FP8_BLOCK":
        tensors = {
            name: torch.randn(128, 128) if layout.quantizes(name) else value
            for name, value in tensors.items()
        }
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, f"{layout.source_prefixes[0]}.obsolete.weight")

    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, mtp_quant_scheme=scheme
    )

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        output_names = set(file.keys())
    assert f"{quantized_module}{weight_suffix}" in output_names
    output_config = json.loads((destination / "config.json").read_text())
    assert "mtp_group" in output_config["quantization_config"]["config_groups"]


def test_block_fp8_rejects_runtime_incompatible_shape_at_preflight(tmp_path):
    """Nemotron's 1856-wide experts convert, but cannot load as 128-block FP8."""
    config, tensors, module, _ = _nemotron_case()
    layout = _resolve_mtp_layout(config, set(tensors))
    tensors = {
        name: torch.randn(128, 128) if layout.quantizes(name) else value
        for name, value in tensors.items()
    }
    tensors["mtp.layers.1.mixer.experts.0.up_proj.weight"] = torch.randn(1856, 128)
    source = tmp_path / "source"
    _write_source(source, tensors)
    with pytest.raises(ValueError, match="runtime requires aligned blocks"):
        mtp._prepare_mtp_source(str(source), config, None, "FP8_BLOCK")


@pytest.mark.parametrize(
    "case_factory,fused_modules",
    [
        (
            _qwen_case,
            [
                "mtp.layers.0.self_attn.q_proj",
                "mtp.layers.0.self_attn.k_proj",
                "mtp.layers.0.self_attn.v_proj",
            ],
        ),
        (
            _glm5_next_case,
            [
                "model.language_model.layers.5.mlp.experts.0.gate_proj",
                "model.language_model.layers.5.mlp.experts.0.up_proj",
            ],
        ),
        (
            _glm_moe_dsa_case,
            [
                "model.layers.5.self_attn.q_a_proj",
                "model.layers.5.self_attn.kv_a_proj_with_mqa",
            ],
        ),
        (
            _nemotron_case,
            [
                "mtp.layers.0.mixer.q_proj",
                "mtp.layers.0.mixer.k_proj",
                "mtp.layers.0.mixer.v_proj",
            ],
        ),
    ],
)
def test_nvfp4a16_fused_projections_share_global_scale(
    tmp_path, case_factory, fused_modules
):
    """Projection sets fused by vLLM receive one shared NVFP4 global scale."""
    config, tensors, _, _ = case_factory()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    layout = _resolve_mtp_layout(config, set(tensors))
    _write_source(source, tensors)
    _write_destination(destination, f"{layout.source_prefixes[0]}.obsolete.weight")

    _quantize_and_save_mtp_tensors(
        str(source),
        str(destination),
        config,
        mtp_quant_scheme="NVFP4A16",
    )

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        scales = [
            file.get_tensor(f"{module}.weight_global_scale") for module in fused_modules
        ]
    assert all(torch.equal(scales[0], scale) for scale in scales[1:])


@pytest.mark.parametrize(
    "case_factory,target_names,ignore_names",
    [
        (
            _qwen_case,
            [
                "mtp.layers.0.self_attn.qkv_proj",
                "mtp.layers.0.mlp.gate_up_proj",
            ],
            ["mtp.fc"],
        ),
        (
            _glm5_next_case,
            [
                "model.layers.5.mlp.experts.0.gate_proj",
                "model.layers.5.mlp.experts.0.up_proj",
                "model.layers.5.mlp.experts.0.down_proj",
                "model.layers.5.mlp.shared_experts.gate_up_proj",
            ],
            [
                "model.layers.5.eh_proj",
                "model.layers.5.self_attn.fused_qkv_a_proj",
            ],
        ),
        (
            _glm_moe_dsa_case,
            [
                "model.layers.5.self_attn.fused_qkv_a_proj",
                "model.layers.5.self_attn.indexer.wq_b",
                "model.layers.5.mlp.experts.0.gate_proj",
                "model.layers.5.mlp.experts.0.up_proj",
                "model.layers.5.mlp.experts.0.down_proj",
                "model.layers.5.mlp.shared_experts.gate_up_proj",
            ],
            [
                "model.layers.5.eh_proj",
                "model.layers.5.self_attn.indexer.wk_weights_proj",
            ],
        ),
        (
            _nemotron_case,
            [
                "mtp.layers.0.eh_proj",
                "mtp.layers.0.mixer.qkv_proj",
                "mtp.layers.1.mixer.experts.0.gate_proj",
                "mtp.layers.1.mixer.experts.0.up_proj",
                "mtp.layers.1.mixer.experts.0.down_proj",
            ],
            ["mtp.layers.1.mixer.gate"],
        ),
    ],
)
def test_runtime_patterns_match_vllm_modules(case_factory, target_names, ignore_names):
    """Architecture policies use the module prefixes constructed by vLLM."""
    config, tensors, _, _ = case_factory()
    layout = _resolve_mtp_layout(config, set(tensors))

    for name in target_names:
        assert any(re.fullmatch(pattern[3:], name) for pattern in layout.targets)
    for name in ignore_names:
        assert any(re.fullmatch(pattern[3:], name) for pattern in layout.ignores)


def test_glm5_next_mtp_keeps_attention_dense():
    """GLM-5.3-Flash MLA stays dense because vLLM constructs it that way."""
    config, tensors, _, _ = _glm5_next_case()
    layout = _resolve_mtp_layout(config, set(tensors))

    quantized, dense = _partition_mtp_tensors(tensors, layout)

    prefix = "model.language_model.layers.5"
    assert f"{prefix}.mlp.experts.0.gate_proj.weight" in quantized
    assert f"{prefix}.self_attn.q_a_proj.weight" in dense
    assert f"{prefix}.self_attn.indexer.wq_b.weight" in dense


def test_glm5_next_validation_ignores_backbone_layers():
    """Only layer ids beyond the Flash backbone belong to MTP."""
    config, tensors, _, _ = _glm5_next_case()
    names = set(tensors) | {"model.language_model.layers.0.mlp.gate.weight"}

    layout = _resolve_mtp_layout(config, names)

    assert layout.source_prefixes == ("model.language_model.layers.5",)


def test_glm_moe_dsa_quantizes_vllm_supported_attention():
    """GLM DSA keeps only the non-quantized indexer fusion inputs dense."""
    config, tensors, _, _ = _glm_moe_dsa_case()
    layout = _resolve_mtp_layout(config, set(tensors))

    quantized, dense = _partition_mtp_tensors(tensors, layout)

    prefix = "model.layers.5.self_attn"
    assert f"{prefix}.q_a_proj.weight" in quantized
    assert f"{prefix}.indexer.wq_b.weight" in quantized
    assert f"{prefix}.indexer.wk.weight" in dense
    assert f"{prefix}.indexer.weights_proj.weight" in dense


def test_glm5_next_discards_base_only_mhc_tensors(tmp_path):
    """Flash MTP omits mHC tensors that vLLM does not construct for MTP."""
    config, tensors, _, _ = _glm5_next_case()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "model.language_model.layers.5.obsolete")

    _quantize_and_save_mtp_tensors(str(source), str(destination), config)

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        output_names = set(file.keys())
    assert not any(".hc_attn_" in name or ".hc_ffn_" in name for name in output_names)


def test_default_preserves_mtp_and_ignores_runtime_prefix(tmp_path):
    """Omitting mtp_quant_scheme preserves MTP tensors and removes a stale MTP group."""
    config, tensors, quantized_module, _ = _qwen_case()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    output_config = json.loads((destination / "config.json").read_text())
    output_config["quantization_config"]["config_groups"]["mtp_group"] = output_config[
        "quantization_config"
    ]["config_groups"]["group_0"]
    (destination / "config.json").write_text(json.dumps(output_config))

    _quantize_and_save_mtp_tensors(str(source), str(destination), config)

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        output_names = set(file.keys())
    assert f"{quantized_module}.weight" in output_names
    assert f"{quantized_module}.weight_scale" not in output_names

    output_config = json.loads((destination / "config.json").read_text())
    quantization_config = output_config["quantization_config"]
    assert "mtp_group" not in quantization_config["config_groups"]
    assert r"re:^mtp\." in quantization_config["ignore"]


def test_default_handles_null_backbone_ignore_list(tmp_path):
    """MTP preservation accepts compressed configs without explicit ignores."""
    config, tensors, _, _ = _qwen_case()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    output_config = json.loads((destination / "config.json").read_text())
    output_config["quantization_config"]["ignore"] = None
    (destination / "config.json").write_text(json.dumps(output_config))

    _quantize_and_save_mtp_tensors(str(source), str(destination), config)

    output_config = json.loads((destination / "config.json").read_text())
    assert r"re:^mtp\." in output_config["quantization_config"]["ignore"]


@pytest.mark.parametrize("activation_scheme", ["dynamic", "static"])
def test_unquantized_native_fp8_mtp(tmp_path, activation_scheme):
    """Explicit BF16 saves dequantized FP8 weights."""
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = {
        "quant_method": "fp8",
        "activation_scheme": activation_scheme,
    }
    tensors["mtp.counter"] = torch.tensor([3], dtype=torch.int64)
    weight_name = "mtp.layers.0.self_attn.q_proj.weight"
    scale_name = "mtp.layers.0.self_attn.q_proj.weight_scale_inv"
    tensors[weight_name] = torch.full((30, 30), 2.0).to(torch.float8_e4m3fn)
    tensors[scale_name] = torch.full((1, 1), 3.0)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, mtp_dequantize=True
    )
    saved = load_file(destination / "model_mtp.safetensors")
    assert saved.keys() == tensors.keys() - {scale_name}
    assert torch.equal(saved[weight_name], torch.full((30, 30), 6.0))
    assert saved[weight_name].dtype == torch.bfloat16
    for name, value in tensors.items():
        if name not in (weight_name, scale_name):
            expected = value.bfloat16() if value.is_floating_point() else value
            assert saved[name].dtype == expected.dtype
            assert torch.equal(saved[name], expected)
    metadata = json.loads((destination / "config.json").read_text())
    assert "mtp_group" not in metadata["quantization_config"]["config_groups"]
    assert r"re:^mtp\." in metadata["quantization_config"]["ignore"]
    with safe_open(source / "model_mtp.safetensors", framework="pt") as file:
        assert file.get_tensor(weight_name).dtype == torch.float8_e4m3fn
        assert scale_name in file.keys()


def test_unquantized_mtp_rejects_fp8_without_scales(tmp_path):
    config, tensors, _, _ = _qwen_case()
    name = "mtp.layers.0.self_attn.q_proj.weight"
    tensors[name] = tensors[name].to(torch.float8_e4m3fn)
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    with pytest.raises(ValueError, match="Cannot reproduce the source MTP format"):
        _quantize_and_save_mtp_tensors(str(source), str(destination), config)
    assert not (destination / "model_mtp.safetensors").exists()


@pytest.mark.parametrize("scheme", [None, "FP8_BLOCK", "FP8"])
def test_native_fp8_reproduces_block_format_through_converter(tmp_path, scheme):
    """Native block-FP8 is dequantized and requantized to the same block format."""
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = {"quant_method": "fp8"}
    layout = _resolve_mtp_layout(config, set(tensors))
    for name in list(tensors):
        if layout.quantizes(name):
            tensors[name] = torch.full((128, 256), 2.0).to(torch.float8_e4m3fn)
            tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.tensor(
                [[3.0, 5.0]]
            )
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    _quantize_and_save_mtp_tensors(str(source), str(destination), config, scheme)
    saved = load_file(destination / "model_mtp.safetensors")
    for name in tensors:
        if not layout.quantizes(name):
            continue
        module = name.removesuffix(".weight")
        assert saved[f"{module}.weight"].dtype == torch.float8_e4m3fn
        assert f"{module}.weight_scale" in saved
        assert f"{module}.weight_scale_inv" not in saved
    output_config = json.loads((destination / "config.json").read_text())[
        "quantization_config"
    ]
    group = output_config["config_groups"]["mtp_group"]
    assert group["weights"]["block_structure"] == [128, 128]
    assert group["input_activations"]["dynamic"] is True
    assert group["format"] == "float-quantized"
    assert r"re:^mtp\." not in output_config["ignore"]


@pytest.mark.parametrize("invalid", ["missing_scale", "static"])
def test_native_fp8_source_rejected_when_unreproducible(tmp_path, invalid):
    """Sources the converter cannot reproduce fail loudly instead of downgrading."""
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = {"quant_method": "fp8"}
    layout = _resolve_mtp_layout(config, set(tensors))
    for name in list(tensors):
        if layout.quantizes(name):
            tensors[name] = torch.ones(128, 128).to(torch.float8_e4m3fn)
            tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.ones(
                1, 1
            )
    module = "mtp.layers.0.self_attn.q_proj"
    if invalid == "missing_scale":
        del tensors[f"{module}.weight_scale_inv"]
    else:
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "static",
        }
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    with pytest.raises(ValueError, match="MTP"):
        _quantize_and_save_mtp_tensors(str(source), str(destination), config)
    assert not (destination / "model_mtp.safetensors").exists()


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("error_type", [ValueError, NotImplementedError])
def test_failed_quantization_is_fatal_without_dequantize(
    tmp_path, monkeypatch, case_name, error_type
):
    """A conversion failure is fatal unless BF16 dequantization was requested."""
    config, tensors, _, _ = CASES[case_name]()
    layout = _resolve_mtp_layout(config, set(tensors))

    def fail(*args):
        raise error_type("injected conversion failure")

    monkeypatch.setattr(mtp, "_compress_mtp_weights", fail)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, f"{layout.source_prefixes[0]}.obsolete.weight")

    with pytest.raises(error_type, match="injected conversion failure"):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, mtp_quant_scheme="NVFP4A16"
        )
    assert not (destination / "model_mtp.safetensors").exists()


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("scheme", [None, "NVFP4A16"])
def test_unknown_projection_is_rejected_at_preflight(tmp_path, case_name, scheme):
    """Both preserve and quantize paths reject incompatible layouts up front."""
    config, tensors, _, _ = CASES[case_name]()
    layout = _resolve_mtp_layout(config, set(tensors))
    tensors[f"{layout.source_prefixes[0]}.unknown_projection.weight"] = torch.randn(
        32, 32
    )
    source = tmp_path / "source"
    _write_source(source, tensors)

    with pytest.raises(ValueError, match="Unsupported MTP projection"):
        mtp._prepare_mtp_source(str(source), config, None, scheme)


def test_layout_requires_the_configured_physical_layers():
    """Nemotron hybrid MTP must include every physical layer in its pattern."""
    config, tensors, _, _ = _nemotron_case()
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if not name.startswith("mtp.layers.1.")
    }

    with pytest.raises(ValueError, match=r"expected \[0, 1\], found \[0\]"):
        _resolve_mtp_layout(config, set(tensors))


def test_unknown_architecture_is_rejected():
    """MTP is never guessed for an architecture without a registered layout."""
    config = PretrainedConfig(
        architectures=["FutureMtpForCausalLM"],
        num_nextn_predict_layers=1,
    )

    with pytest.raises(ValueError, match="FutureMtpForCausalLM"):
        _resolve_mtp_layout(config, {"mtp.layers.0.q_proj.weight"})


def test_mtp_model_requires_source_checkpoint():
    """MTP preservation fails clearly when source tensors cannot be located."""
    model = SimpleNamespace(
        config=PretrainedConfig(
            architectures=["Qwen3_5ForConditionalGeneration"],
            mtp_num_hidden_layers=1,
        ),
        name_or_path="",
    )

    with pytest.raises(ValueError, match="no source checkpoint path"):
        save_mtp_tensors(model, "unused")


@pytest.mark.parametrize(
    "quantization_config,block_size",
    [
        ({"weight_block_size": [2, 2]}, 2),
        ({"weight_block_size": None}, 128),
        ({}, 128),
    ],
    ids=["explicit", "null", "missing"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_fp8_blocks_are_dequantized_before_requantization(
    quantization_config, block_size, device
):
    """Native block-FP8 MTP weights are restored before applying mtp_quant_scheme."""
    if device == "cuda" and (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("Requires a CUDA accelerator")
    weight = torch.full((2 * block_size, 2 * block_size), 2.0, device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    tensors = {
        "mtp.layers.0.self_attn.q_proj.weight": weight,
        "mtp.layers.0.self_attn.q_proj.weight_scale_inv": scales,
    }
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8", **quantization_config}
    )

    output = _dequantize_fp8_blocks(tensors, config)

    assert "mtp.layers.0.self_attn.q_proj.weight_scale_inv" not in output
    assert output["mtp.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    expected = 2 * scales.repeat_interleave(block_size, 0).repeat_interleave(
        block_size, 1
    )
    assert torch.equal(output["mtp.layers.0.self_attn.q_proj.weight"], expected)


def test_missing_local_mtp_shard_is_fatal(tmp_path):
    """A partial source checkpoint cannot silently produce a partial output."""
    config, tensors, _, _ = _qwen_case()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    weight_map = {name: "missing.safetensors" for name in tensors}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    _write_destination(destination, "mtp.obsolete.weight")

    with pytest.raises(FileNotFoundError, match="MTP shard not found"):
        _quantize_and_save_mtp_tensors(str(source), str(destination), config)


def test_source_index_uses_requested_revision(tmp_path, monkeypatch):
    """Remote MTP discovery uses the same revision as the backbone load."""
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.layers.0.self_attn.q_proj.weight": "model.safetensors"
                }
            }
        )
    )
    calls = []

    def download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(index)

    monkeypatch.setattr(mtp, "hf_hub_download", download)

    weight_map, _, local = mtp._source_weight_map("org/model", "commit")

    assert not local
    assert weight_map["mtp.layers.0.self_attn.q_proj.weight"] == ("model.safetensors")
    assert calls == [
        ("org/model", "model.safetensors.index.json", "commit"),
    ]


@pytest.mark.parametrize(
    "alias",
    ["BF16", "bf16", "bfloat16", "none", "dense", "unquantized"],
)
def test_unquantized_scheme_aliases_are_rejected(alias):
    """Only None and the explicit BF16 preset select non-quantizing paths."""
    with pytest.raises(ValueError, match="mtp_quant_scheme=None"):
        _resolve_mtp_scheme(alias)


def test_none_selects_source_preservation():
    assert _resolve_mtp_scheme(None) is None


def test_scheme_keeps_calibration_free_activations():
    """Dynamic activation quantization remains enabled."""
    assert _resolve_mtp_scheme("FP8_DYNAMIC").input_activations.dynamic is True
    assert _resolve_mtp_scheme("MXFP4").input_activations.dynamic is True
    assert _resolve_mtp_scheme("NVFP4A16").input_activations is None


@pytest.mark.parametrize("scheme", ["NVFP4", "FP8"])
def test_calibration_dependent_scheme_preserves_source(scheme):
    """Calibration-dependent MTP schemes are deferred to a future pathway."""
    assert _resolve_mtp_scheme(scheme) is None


def test_scheme_rejects_invalid_type():
    """mtp_quant_scheme accepts only preset names, scheme objects, or None."""
    with pytest.raises(TypeError, match="mtp_quant_scheme must be"):
        _resolve_mtp_scheme(123)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("scheme", ["NVFP4A16", "FP8_DYNAMIC", "FP8_BLOCK", "MXFP4"])
@pytest.mark.parametrize("save_scheme", [None, "same", "BF16"])
def test_quantized_source_round_trip_and_bf16(tmp_path, case_name, scheme, save_scheme):
    config, tensors, _, _ = CASES[case_name]()
    layout = _resolve_mtp_layout(config, set(tensors))
    if scheme == "FP8_BLOCK":
        tensors = {
            name: torch.randn(128, 128) if layout.quantizes(name) else value
            for name, value in tensors.items()
        }
    source, first, second = (tmp_path / name for name in ("source", "first", "second"))
    _write_source(source, tensors)
    for destination in (first, second):
        _write_destination(destination, f"{layout.source_prefixes[0]}.obsolete.weight")
    _quantize_and_save_mtp_tensors(str(source), str(first), config, scheme)

    _quantize_and_save_mtp_tensors(
        str(first),
        str(second),
        config,
        scheme if save_scheme == "same" else None,
        mtp_dequantize=save_scheme == "BF16",
    )
    original = load_file(first / "model_mtp.safetensors")
    saved = load_file(second / "model_mtp.safetensors")
    first_config = json.loads((first / "config.json").read_text())[
        "quantization_config"
    ]
    second_config = json.loads((second / "config.json").read_text())[
        "quantization_config"
    ]
    if save_scheme != "BF16":
        # None and an explicit matching scheme both reproduce the source format
        # through the converter (identical settings and weight storage; scale
        # precision may narrow to BF16 through the intermediate dequantization).
        assert saved.keys() == original.keys()
        for name in original:
            if name.rpartition(".")[-1] in ("weight", "weight_packed"):
                assert saved[name].dtype == original[name].dtype
        assert (
            second_config["config_groups"]["mtp_group"]
            == first_config["config_groups"]["mtp_group"]
        )
        return
    assert saved.keys() == {name for name in tensors if not layout.discards(name)}
    source_scheme = mtp.QuantizationScheme.model_validate(
        first_config["config_groups"]["mtp_group"]
    )
    compressor = BaseCompressor.get_value_from_registry(source_scheme.format)
    for name in saved:
        if layout.quantizes(name):
            module = name.removesuffix(".weight")
            state = {
                param: original[f"{module}.{param}"]
                for param in compressor.compression_param_names(source_scheme)
            }
            expected = compressor.decompress(state, source_scheme)["weight"].bfloat16()
            assert saved[name].dtype == torch.bfloat16
            assert saved[name].shape == tensors[name].shape
            assert torch.equal(saved[name], expected)
        else:
            assert torch.equal(saved[name], tensors[name].bfloat16())
    assert "mtp_group" not in second_config["config_groups"]
    assert (
        second_config["config_groups"]["group_0"]
        == first_config["config_groups"]["group_0"]
    )
    assert set(layout.full_precision_ignores()) <= set(second_config["ignore"])


@pytest.mark.parametrize("failure_point", ["dequantizer", "converter"])
def test_native_fp8_failure_does_not_mutate_source(
    tmp_path, monkeypatch, failure_point
):
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    layout = _resolve_mtp_layout(config, set(tensors))
    for name in list(tensors):
        if layout.quantizes(name):
            tensors[name] = torch.full((32, 32), 2.0).to(torch.float8_e4m3fn)
            tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.full(
                (1, 1), 3.0
            )
    original = {name: value.clone() for name, value in tensors.items()}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    monkeypatch.setattr(mtp, "_load_mtp_tensors", lambda *args: tensors)
    if failure_point == "dequantizer":
        dequantize = mtp._dequantize_fp8_blocks

        def fail(partial, config):
            # One module was replaced before the dequantizer failed on the next.
            module = "mtp.layers.0.self_attn.q_proj"
            converted = dequantize(
                {
                    key: value
                    for key, value in partial.items()
                    if key.startswith(module)
                },
                config,
            )
            partial.pop(f"{module}.weight_scale_inv")
            partial.update(converted)
            raise ValueError("injected midway failure")

        monkeypatch.setattr(mtp, "_dequantize_fp8_blocks", fail)
    else:

        def fail(*args):
            raise ValueError("injected after dequantization")

        monkeypatch.setattr(mtp, "_compress_mtp_weights", fail)
    expected_error = (
        "injected midway failure"
        if failure_point == "dequantizer"
        else "injected after dequantization"
    )
    with pytest.raises(ValueError, match=expected_error):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, "NVFP4A16"
        )
    assert not (destination / "model_mtp.safetensors").exists()
    assert tensors.keys() == original.keys()
    for name in original:
        assert tensors[name].dtype == original[name].dtype
        assert torch.equal(tensors[name].float(), original[name].float())


@pytest.mark.parametrize("architecture", list(mtp._MTP_LAYOUTS))
def test_registered_architecture_aliases(architecture):
    factory = mtp._MTP_LAYOUTS[architecture]
    case = {
        mtp._qwen3_5_layout: _qwen_case,
        mtp._glm5_next_layout: _glm5_next_case,
        mtp._glm_moe_dsa_layout: _glm_moe_dsa_case,
        mtp._nemotron_h_layout: _nemotron_case,
    }[factory]
    config, tensors, _, _ = case()
    config.architectures = [architecture]
    layout = _resolve_mtp_layout(config, set(tensors))
    assert all(layout.owns(name) for name in tensors)
    assert bool(layout.targets) == (not architecture.startswith("Qwen3_5Moe"))


@pytest.mark.parametrize(
    "architecture", ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]
)
@pytest.mark.parametrize("scheme", [None, "NVFP4A16"])
def test_qwen_moe_preserves_without_claiming_quantization(
    tmp_path, architecture, scheme
):
    config, tensors, _, _ = _qwen_case()
    config.architectures = [architecture]
    tensors["mtp.layers.0.mlp.experts.0.gate_proj.weight"] = torch.randn(32, 32)
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    _quantize_and_save_mtp_tensors(str(source), str(destination), config, scheme)
    saved = load_file(destination / "model_mtp.safetensors")
    assert saved.keys() == tensors.keys()
    assert all(torch.equal(saved[name], value) for name, value in tensors.items())
    groups = json.loads((destination / "config.json").read_text())[
        "quantization_config"
    ]["config_groups"]
    assert "mtp_group" not in groups


@pytest.mark.parametrize("invalid", ["output_activations", "weights"])
def test_scheme_validation_does_not_mutate_caller(invalid):
    scheme = preset_name_to_scheme("FP8_DYNAMIC", ["Linear"])
    if invalid == "weights":
        scheme.weights = None
    else:
        scheme.output_activations = scheme.input_activations.model_copy(
            update={"dynamic": False}
        )
    original = scheme.model_copy(deep=True)
    assert _resolve_mtp_scheme(scheme) is None
    assert scheme == original


@pytest.mark.parametrize(
    "ignore", ["re:^mtp.*", "re:.*q_proj", "mtp.layers.0.self_attn.qkv_proj"]
)
def test_quantized_mtp_cannot_be_ignored(tmp_path, ignore):
    config, tensors, _, _ = _qwen_case()
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    metadata = json.loads((destination / "config.json").read_text())
    metadata["quantization_config"]["ignore"].append(ignore)
    (destination / "config.json").write_text(json.dumps(metadata))
    if ignore == "re:^mtp.*":
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, "NVFP4A16"
        )
        assert (
            ignore
            not in json.loads((destination / "config.json").read_text())[
                "quantization_config"
            ]["ignore"]
        )
    else:
        with pytest.raises(ValueError, match="conflicts with ignore list"):
            _quantize_and_save_mtp_tensors(
                str(source), str(destination), config, "NVFP4A16"
            )


def test_write_failure_is_not_conversion_fallback(tmp_path, monkeypatch):
    config, tensors, _, _ = _qwen_case()
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    calls = []

    def fail(*args):
        calls.append(args)
        raise ValueError("injected write failure")

    monkeypatch.setattr(mtp, "save_file", fail)
    with pytest.raises(ValueError, match="injected write failure"):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, "NVFP4A16"
        )
    assert len(calls) == 1


def test_native_reuse_probe_does_not_block_conversion(tmp_path, monkeypatch):
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = {"quant_method": "fp8", "activation_scheme": "static"}
    module = "mtp.layers.0.self_attn.q_proj"
    tensors[f"{module}.weight"] = torch.ones(32, 32).to(torch.float8_e4m3fn)
    tensors[f"{module}.weight_scale_inv"] = torch.ones(1, 1)
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    def reached(*args):
        raise ValueError("reached existing dequantizer")

    monkeypatch.setattr(mtp, "_dequantize_mtp_tensors", reached)
    with pytest.raises(ValueError, match="reached existing dequantizer"):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, "NVFP4A16"
        )


@pytest.mark.parametrize("revision", [None, "main", "another-branch"])
def test_mtp_uses_loaded_backbone_commit(monkeypatch, revision):
    config, _, _, _ = _qwen_case()
    config._commit_hash = "loaded-commit"
    model = SimpleNamespace(config=config, name_or_path="owner/model")
    prepare = Mock(return_value="prepared")
    monkeypatch.setattr(mtp, "_prepare_mtp_source", prepare)
    assert mtp.prepare_mtp_save(model, revision=revision) == "prepared"
    assert prepare.call_args.args[2] == "loaded-commit"


def test_mtp_revision_fallback_without_commit(monkeypatch):
    config, _, _, _ = _qwen_case()
    config._commit_hash = None
    prepare = Mock(return_value="prepared")
    monkeypatch.setattr(mtp, "_prepare_mtp_source", prepare)
    mtp.prepare_mtp_save(
        SimpleNamespace(config=config, name_or_path="owner/model"), revision="tag"
    )
    assert prepare.call_args.args[2] == "tag"


def test_unsupported_architecture_skips_mtp_before_source_access(monkeypatch):
    config = PretrainedConfig(
        architectures=["FutureMtpForCausalLM"], num_nextn_predict_layers=1
    )
    prepare = Mock(side_effect=AssertionError("Unsupported MTP must not access source"))
    monkeypatch.setattr(mtp, "_prepare_mtp_source", prepare)
    assert mtp.prepare_mtp_save(SimpleNamespace(config=config)) is None
    prepare.assert_not_called()


@pytest.mark.parametrize("scheme", [None, "FP8_DYNAMIC"])
@pytest.mark.parametrize("dequantize", [False, True])
def test_mtp_quantization_and_dequantization_controls(tmp_path, scheme, dequantize):
    config, tensors, _, _ = _qwen_case()
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, scheme, mtp_dequantize=dequantize
    )
    saved = load_file(destination / "model_mtp.safetensors")
    name = "mtp.layers.0.self_attn.q_proj.weight"
    if scheme:
        assert saved[name].dtype == torch.float8_e4m3fn
        assert name.removesuffix(".weight") + ".weight_scale" in saved
    else:
        expected = tensors[name].bfloat16() if dequantize else tensors[name]
        assert saved[name].dtype == expected.dtype
        assert torch.equal(saved[name], expected)


@pytest.mark.parametrize(
    "error_type,wrapped",
    [
        (RuntimeError, False),
        (TypeError, False),
        (AttributeError, False),
        (KeyError, False),
        (MemoryError, False),
        (torch.OutOfMemoryError, False),
        (MemoryError, True),
        (torch.OutOfMemoryError, True),
    ],
)
def test_unexpected_conversion_errors_are_fatal(
    tmp_path, monkeypatch, error_type, wrapped
):
    config, tensors, _, _ = _qwen_case()
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    def fail(*args):
        if wrapped:
            try:
                raise error_type("injected failure")
            except error_type as error:
                raise ValueError("wrapped conversion failure") from error
        raise error_type("injected failure")

    monkeypatch.setattr(mtp, "_compress_mtp_weights", fail)
    with pytest.raises(error_type, match="injected failure"):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, "FP8_DYNAMIC"
        )
    assert not (destination / "model_mtp.safetensors").exists()


@pytest.mark.parametrize("dequantize", [False, True])
def test_conversion_fallback_honors_dequantize(tmp_path, monkeypatch, dequantize):
    config, tensors, _, _ = _qwen_case()
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    monkeypatch.setattr(
        mtp,
        "_compress_mtp_weights",
        Mock(side_effect=ValueError("unsupported conversion")),
    )
    if not dequantize:
        with pytest.raises(ValueError, match="unsupported conversion"):
            _quantize_and_save_mtp_tensors(
                str(source), str(destination), config, "FP8_DYNAMIC"
            )
        assert not (destination / "model_mtp.safetensors").exists()
        return
    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, "FP8_DYNAMIC", mtp_dequantize=True
    )
    saved = load_file(destination / "model_mtp.safetensors")
    for name, tensor in tensors.items():
        expected = tensor.bfloat16() if tensor.is_floating_point() else tensor
        assert saved[name].dtype == expected.dtype
        assert torch.equal(saved[name], expected)
    groups = json.loads((destination / "config.json").read_text())[
        "quantization_config"
    ]["config_groups"]
    assert "mtp_group" not in groups


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"quant_method": "other"},
        *(
            {"quant_method": "fp8", "weight_block_size": block}
            for block in (128, 0, [], [128], [128, 0], [128, True], [128, 1.5])
        ),
    ],
)
@pytest.mark.parametrize("dequantize", [False, True])
def test_native_fp8_metadata_rejected_before_save(tmp_path, metadata, dequantize):
    config, tensors, _, _ = _qwen_case()
    config.quantization_config = metadata
    name = "mtp.layers.0.self_attn.q_proj.weight"
    tensors[name] = tensors[name].to(torch.float8_e4m3fn)
    tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.ones(1, 1)
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")
    with pytest.raises(ValueError, match="Native FP8"):
        _quantize_and_save_mtp_tensors(
            str(source), str(destination), config, mtp_dequantize=dequantize
        )
    with pytest.raises(ValueError, match="Native FP8"):
        _dequantize_fp8_blocks(tensors, config)
    assert not (destination / "model_mtp.safetensors").exists()


def test_calibration_dependent_source_scheme_is_rejected(tmp_path):
    """Regression: a source scheme needing calibration cannot be reproduced
    data-free through the converter, so default (None) saving must fail loudly
    rather than emit zeroed activation scales."""
    config, tensors, _, _ = _qwen_case()
    source, first, second = (tmp_path / name for name in ("source", "first", "second"))
    _write_source(source, tensors)
    for destination in (first, second):
        _write_destination(destination, "mtp.obsolete.weight")
    _quantize_and_save_mtp_tensors(str(source), str(first), config, "FP8_DYNAMIC")
    config_path = first / "config.json"
    saved_config = json.loads(config_path.read_text())
    group = saved_config["quantization_config"]["config_groups"]["mtp_group"]
    group["input_activations"]["strategy"] = "tensor"
    group["input_activations"]["dynamic"] = False
    config_path.write_text(json.dumps(saved_config))
    with pytest.raises(ValueError, match="requires calibration"):
        _quantize_and_save_mtp_tensors(str(first), str(second), config)
    assert not (second / "model_mtp.safetensors").exists()


def test_native_fp8_expert_block_alignment_checked_without_scheme(tmp_path):
    """Regression: default (None) saving reproduces the source block scheme, so
    preflight must reject block-misaligned experts exactly as an explicit
    FP8_BLOCK request would."""
    config, tensors, _, _ = _glm5_next_case()
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    layout = _resolve_mtp_layout(config, set(tensors))
    for name in list(tensors):
        if ".experts." in name and layout.quantizes(name):
            tensors[name] = torch.ones(128, 192).to(torch.float8_e4m3fn)
            tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.ones(
                1, 2
            )
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, f"{layout.source_prefixes[0]}.obsolete.weight")
    with pytest.raises(ValueError, match="not divisible"):
        _quantize_and_save_mtp_tensors(str(source), str(destination), config)
    assert not (destination / "model_mtp.safetensors").exists()


def test_native_fp8_scale_grid_shape_is_validated():
    """Regression: a malformed scale grid must be rejected rather than silently
    broadcast by the block dequantizer."""
    module = "mtp.layers.0.self_attn.q_proj"
    tensors = {
        f"{module}.weight": torch.ones(256, 256).to(torch.float8_e4m3fn),
        f"{module}.weight_scale_inv": torch.ones(1, 2),
    }
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
    )
    with pytest.raises(ValueError, match="scale grid"):
        _dequantize_fp8_blocks(tensors, config)
