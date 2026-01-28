import json
import os
from typing import Optional

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from loguru import logger

from .helpers import find_config_path, find_safetensors_index_path

__all__ = ["update_config", "update_safetensors_index"]

# Layer name patterns for detecting which config fields to update
_MLP_INTERMEDIATE_PATTERNS = ("gate_proj", "up_proj", "down_proj")
_MOE_INTERMEDIATE_PATTERNS = ("experts.",)


def _infer_padded_dimensions(
    padded_layers: dict[str, dict[str, list[int]]],
) -> dict[str, int]:
    """
    Infer which model config dimensions need to be updated based on padded layers.

    Analyzes the padded layer names and their new shapes to determine the correct
    padded dimension values for config fields like intermediate_size.

    :param padded_layers: Dict mapping module names to their padding info.
        Format: {"module_name": {"original_shape": [out, in], "padded_shape": [out, in]}}
    :return: Dict of config field names to their padded values
    """
    updates = {}

    for module_name, info in padded_layers.items():
        padded_shape = info.get("padded_shape", [])
        if len(padded_shape) != 2:
            continue

        padded_out, padded_in = padded_shape

        # Check if this is an MLP layer that affects intermediate_size
        is_moe = any(pattern in module_name for pattern in _MOE_INTERMEDIATE_PATTERNS)
        is_mlp = any(pattern in module_name for pattern in _MLP_INTERMEDIATE_PATTERNS)

        if is_mlp:
            # gate_proj/up_proj: out_features = intermediate_size
            # down_proj: in_features = intermediate_size
            if "gate_proj" in module_name or "up_proj" in module_name:
                intermediate_size = padded_out
            elif "down_proj" in module_name:
                intermediate_size = padded_in
            else:
                continue

            config_key = (
                "moe_intermediate_size" if is_moe else "intermediate_size"
            )

            # Use the largest padded value if multiple layers are padded
            if config_key not in updates or intermediate_size > updates[config_key]:
                updates[config_key] = intermediate_size

    return updates


def update_config(
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
    padded_layers: Optional[dict[str, dict[str, list[int]]]] = None,
):
    """
    Update the config.json file with quantization configuration.

    When block quantization requires padding weights to divisible dimensions,
    this function also updates the model's config dimensions (e.g., intermediate_size)
    to match the padded weights. This ensures inference frameworks like vLLM can
    correctly load and use the quantized model without needing special handling.

    :param save_directory: Directory containing the config.json file
    :param scheme_name: Name of the quantization scheme
    :param scheme: The quantization scheme
    :param ignore: List of modules to ignore
    :param padded_layers: Optional dict mapping module names to their padding info.
        Format: {"module_name": {"original_shape": [out, in], "padded_shape": [out, in]}}
    """
    # construct quantization config
    qconfig = QuantizationConfig.model_validate(
        {
            "config_groups": {scheme_name: scheme},
            "ignore": ignore,
            "quantization_status": QuantizationStatus.COMPRESSED,
        }
    )

    # construct compression (quantization) config
    qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME, "format"])
    qconfig_data = {
        COMPRESSION_VERSION_NAME: ct_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
        SPARSITY_CONFIG_NAME: {},
        TRANSFORM_CONFIG_NAME: {},
        "format": scheme.format,
        **qconfig_data,
    }

    # write results to config.json file
    config_file_path = find_config_path(save_directory)
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data

        # Update model config dimensions if layers were padded
        if padded_layers:
            dimension_updates = _infer_padded_dimensions(padded_layers)
            for key, value in dimension_updates.items():
                if key in config_data:
                    logger.info(
                        f"Updating {key} from {config_data[key]} to {value} "
                        "due to block quantization padding"
                    )
                    config_data[key] = value

        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)

    else:
        logger.warning(
            f"Could not find config file in {save_directory}. "
            f"Please {json.dumps(qconfig_data, indent=2, sort_keys=True)}"
        )


def update_safetensors_index(
    save_directory: str | os.PathLike,
    total_size: int,
    weight_map: dict[str, str],
):
    file_path = find_safetensors_index_path(save_directory)
    if file_path is None:
        return

    with open(file_path, "w") as file:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                },
                "weight_map": weight_map,
            },
            file,
            indent=2,
            sort_keys=True,
        )
