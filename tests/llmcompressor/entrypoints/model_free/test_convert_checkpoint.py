import json

import pytest
from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    FP8BlockDequantizer,
    ModelOptNvfp4Converter,
    convert_checkpoint,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationMetadata,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import NVFP4

from tests.testing_utils import requires_cadence

# NOTE: This file contains tests for compressed_tensors.entrypoints.convert
# that are either long-running or involve larger models. They have been placed
# here to leverage llm-compressor's nightly testing CI/CD.


@requires_cadence("nightly")
def test_convert_nvfp4_checkpoint(tmp_path):
    """
    Test that compressed-tensors convert_checkpoint entrypoint
    can be run on a pre-existing modelopt checkpoint
    """
    MODEL_ID = "nvidia/Qwen3-8B-NVFP4"
    convert_outdir = tmp_path / "convert_out"

    right_targets = [
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(q|k|v|o)_proj$",
    ]
    wrong_targets = [
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(q|k|o)_proj$",
    ]
    right_kv_cache_scheme = QuantizationArgs(
        num_bits=8, dynamic=False, type=QuantizationType.FLOAT
    )

    with pytest.raises(ValueError):
        convert_checkpoint(
            model_stub=MODEL_ID,
            save_directory=convert_outdir,
            converter=ModelOptNvfp4Converter(
                targets=wrong_targets,
                kv_cache_scheme=right_kv_cache_scheme,
            ),
        )

    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=convert_outdir,
        converter=ModelOptNvfp4Converter(
            targets=right_targets,
            kv_cache_scheme=right_kv_cache_scheme,
        ),
    )

    with open(convert_outdir / "config.json", "r") as f:
        config = json.load(f)

        qconfig = QuantizationConfig.model_validate(config["quantization_config"])

    assert qconfig.format == "nvfp4-pack-quantized"
    assert qconfig.quant_method == "compressed-tensors"
    assert len(qconfig.config_groups) == 1
    # assert weights and input_activations are a superset of what's in the NVFP4 preset
    assert (
        qconfig.config_groups["config_group_0"].weights.model_dump().items()
        >= NVFP4["weights"].model_dump().items()
    )
    assert (
        qconfig.config_groups["config_group_0"].input_activations.model_dump().items()
        >= NVFP4["input_activations"].model_dump().items()
    )

    with open(convert_outdir / "model.safetensors.index.json", "r") as f:
        allowed_suffixes = [
            "weight",
            "weight_scale",
            "weight_packed",
            "weight_global_scale",
            "input_global_scale",
            "k_scale",
            "v_scale",
        ]
        data = json.load(f)
        keys = data["weight_map"].keys()
        for key in keys:
            assert any(
                key.endswith(suffix) for suffix in allowed_suffixes
            ), f"Unexpected key found: {key}"


@requires_cadence("nightly")
def test_dequantize_fp8block_checkpoint(tmp_path):
    """
    Test that compressed-tensors convert_checkpoint entrypoint
    can convert FP8 block-quantized checkpoints back to bfloat16
    """
    MODEL_ID = "qwen-community/Qwen3-4B-FP8"
    convert_outdir = tmp_path / "convert_out"

    right_targets = [
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(q|k|v|o)_proj$",
    ]
    wrong_targets = [
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(q|k|o)_proj$",  # missing v_proj
    ]

    # Test that wrong targets raise ValueError during validation
    with pytest.raises(ValueError):
        convert_checkpoint(
            model_stub=MODEL_ID,
            save_directory=convert_outdir,
            converter=FP8BlockDequantizer(
                targets=wrong_targets,
                weight_block_size=(128, 128),
            ),
        )

    # Convert Qwen3-4B-FP8 back to dense bfloat16 format
    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=convert_outdir,
        converter=FP8BlockDequantizer(
            targets=right_targets,
            weight_block_size=(128, 128),
        ),
    )

    # Validate output config.json
    with open(convert_outdir / "config.json", "r") as f:
        config = json.load(f)

        # FP8BlockToBfloat16Converter removes quantization, so no quantization_config
        assert "quantization_config" not in config

    # Validate output safetensors index
    with open(convert_outdir / "model.safetensors.index.json", "r") as f:
        allowed_suffixes = [
            "weight",  # All weights should be converted to bfloat16
        ]
        disallowed_suffixes = [
            "weight_scale",
            "weight_scale_inv",  # These should be removed during conversion
        ]
        data = json.load(f)
        keys = data["weight_map"].keys()
        for key in keys:
            assert any(
                key.endswith(suffix) for suffix in allowed_suffixes
            ), f"Unexpected key found: {key}"
            assert not any(
                key.endswith(suffix) for suffix in disallowed_suffixes
            ), f"Found disallowed key (should have been removed): {key}"


@requires_cadence("nightly")
@pytest.mark.parametrize(
    "model_id",
    (
        "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
        "nm-testing/tinyllama-w4a16-compressed",
        "nm-testing/Meta-Llama-3-8B-Instruct-nonuniform-test",
    ),
)
def test_dequantize_ct_checkpoint(model_id, tmp_path):
    """
    Test that compressed-tensors convert_checkpoint entrypoint
    can convert CT checkpoint back to bfloat16
    """
    convert_outdir = tmp_path / "convert_out"

    # Convert Qwen3-4B-FP8 back to dense bfloat16 format
    convert_checkpoint(
        model_stub=model_id,
        save_directory=convert_outdir,
        converter=CompressedTensorsDequantizer(
            model_stub=model_id, ignore=["lm_head", "re:.*embed_tokens$"]
        ),
    )

    # Validate output config.json
    with open(convert_outdir / "config.json", "r") as f:
        config = json.load(f)

        # Dequantizer removes quantization, so no quantization_config
        assert "quantization_config" not in config

    # Validate output safetensors index
    with open(convert_outdir / "model.safetensors.index.json", "r") as f:
        allowed_suffixes = [
            "weight",  # All weights should be converted to bfloat16
        ]
        disallowed_suffixes = QuantizationMetadata.all_qparam_names()
        data = json.load(f)
        keys = data["weight_map"].keys()
        for key in keys:
            assert any(
                key.endswith(suffix) for suffix in allowed_suffixes
            ), f"Unexpected key found: {key}"
            assert not any(
                key.endswith(suffix) for suffix in disallowed_suffixes
            ), f"Found disallowed key (should have been removed): {key}"
