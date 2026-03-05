import json

import pytest
from compressed_tensors.entrypoints.convert import (
    ModelOptNvfp4Converter,
    convert_checkpoint,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import NVFP4

from tests.testing_utils import requires_cadence

# NOTE: This file contains tests for compressed_tensors.entrypoints.convert
# that are either long-running or involve larger models. They have been placed
# here to leverage llm-compressor's nightly testing CI/CD.


@requires_cadence("nightly")
def test_convert_checkpoint(tmp_path):
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
    wrong_kv_cache_scheme = None

    with pytest.raises(ValueError):
        convert_checkpoint(
            model_stub=MODEL_ID,
            save_directory=convert_outdir,
            converter=ModelOptNvfp4Converter(
                targets=right_targets,
                kv_cache_scheme=wrong_kv_cache_scheme,
            ),
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
