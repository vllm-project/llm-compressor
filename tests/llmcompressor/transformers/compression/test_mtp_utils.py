import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    preset_name_to_scheme,
)
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PretrainedConfig

from llmcompressor.transformers.compression import mtp
from llmcompressor.transformers.compression.mtp import (
    _dequantize_fp8_blocks,
    _partition_mtp_tensors,
    _quantize_and_save_mtp_tensors,
    _resolve_mtp_layout,
    _resolve_mtp_scheme,
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
def test_nvfp4_quantizes_supported_architecture_layouts(tmp_path, case_name):
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
        mtp_scheme="NVFP4",
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
        ("MXFP4", ".weight_packed"),
        ("NVFP4", ".weight_packed"),
    ],
)
def test_data_free_mtp_schemes(tmp_path, scheme, weight_suffix):
    """Supported data-free schemes produce an MTP checkpoint."""
    config, tensors, quantized_module, _ = _qwen_case()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, mtp_scheme=scheme
    )

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        output_names = set(file.keys())
    assert f"{quantized_module}{weight_suffix}" in output_names
    output_config = json.loads((destination / "config.json").read_text())
    assert "mtp_group" in output_config["quantization_config"]["config_groups"]


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
def test_nvfp4_fused_projections_share_global_scale(
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
        mtp_scheme="NVFP4",
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
    """Omitting mtp_scheme preserves MTP tensors and removes a stale MTP group."""
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


def test_default_does_not_dequantize_source_mtp(tmp_path):
    """Preserving MTP leaves source-format tensors unchanged."""
    config, tensors, _, _ = _qwen_case()
    weight_name = "mtp.layers.0.self_attn.q_proj.weight"
    scale_name = "mtp.layers.0.self_attn.q_proj.weight_scale_inv"
    tensors[weight_name] = tensors[weight_name].to(torch.float8_e4m3fn)
    tensors[scale_name] = torch.ones(1, 1)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    _quantize_and_save_mtp_tensors(str(source), str(destination), config)

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        assert file.get_tensor(weight_name).dtype == torch.float8_e4m3fn
        assert scale_name in file.keys()


def test_failed_quantization_preserves_source_mtp(tmp_path):
    """A data-free quantization failure never drops supported MTP tensors."""
    config, tensors, _, _ = _qwen_case()
    unsupported = "mtp.layers.0.self_attn.new_proj.weight"
    tensors[unsupported] = torch.randn(32, 32)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_source(source, tensors)
    _write_destination(destination, "mtp.obsolete.weight")

    _quantize_and_save_mtp_tensors(
        str(source), str(destination), config, mtp_scheme="NVFP4"
    )

    with safe_open(destination / "model_mtp.safetensors", framework="pt") as file:
        assert set(file.keys()) == set(tensors)
    output_config = json.loads((destination / "config.json").read_text())
    quantization_config = output_config["quantization_config"]
    assert "mtp_group" not in quantization_config["config_groups"]
    assert r"re:^mtp\." in quantization_config["ignore"]


def test_unknown_projection_is_rejected():
    """A new projection must be added to an architecture layout explicitly."""
    config, tensors, _, _ = _qwen_case()
    tensors["mtp.layers.0.self_attn.new_proj.weight"] = torch.randn(32, 32)
    layout = _resolve_mtp_layout(config, set(tensors))

    with pytest.raises(ValueError, match="Unsupported MTP projection"):
        _partition_mtp_tensors(tensors, layout)


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


def test_native_fp8_blocks_are_dequantized_before_requantization():
    """Native block-FP8 MTP weights are restored before applying mtp_scheme."""
    weight = torch.arange(1, 17).reshape(4, 4).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensors = {
        "mtp.layers.0.self_attn.q_proj.weight": weight,
        "mtp.layers.0.self_attn.q_proj.weight_scale_inv": scales,
    }
    config = PretrainedConfig(quantization_config={"weight_block_size": [2, 2]})

    output = _dequantize_fp8_blocks(tensors, config)

    assert "mtp.layers.0.self_attn.q_proj.weight_scale_inv" not in output
    assert output["mtp.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert output["mtp.layers.0.self_attn.q_proj.weight"][0, 2] == 6
    assert output["mtp.layers.0.self_attn.q_proj.weight"][2, 0] == 27


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
    ["bf16", "BF16", "bfloat16", "none", "dense", "unquantized"],
)
def test_unquantized_scheme_aliases(alias):
    """Full-precision aliases resolve to no MTP quantization scheme."""
    assert _resolve_mtp_scheme(alias) is None


def test_scheme_keeps_only_calibration_free_activations():
    """Static and local-dynamic activations are dropped; dynamic stays enabled."""
    assert _resolve_mtp_scheme("NVFP4").input_activations is None
    assert _resolve_mtp_scheme("FP8").input_activations is None
    assert _resolve_mtp_scheme("FP8_DYNAMIC").input_activations.dynamic is True


def test_scheme_rejects_invalid_type():
    """mtp_scheme accepts only preset names, scheme objects, or None."""
    with pytest.raises(TypeError, match="mtp_scheme must be"):
        _resolve_mtp_scheme(123)
