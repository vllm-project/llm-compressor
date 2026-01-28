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
# These are "safe-to-pad" layers where padding zeros doesn't affect model correctness
# because the extra dimensions will produce zero outputs that can be truncated

# MLP intermediate layers - padding these is safe because:
# - gate_proj/up_proj: extra output channels produce zeros (no input contribution)
# - down_proj: extra input channels receive zeros (no output contribution)
_MLP_GATE_UP_PATTERNS = ("gate_proj", "up_proj", "w1", "w3", "fc1")
_MLP_DOWN_PATTERNS = ("down_proj", "w2", "fc2")
_MLP_INTERMEDIATE_PATTERNS = _MLP_GATE_UP_PATTERNS + _MLP_DOWN_PATTERNS

# MoE layer patterns
_MOE_PATTERNS = ("experts.", "mlp.experts", "block_sparse_moe")

# Config field name aliases for different model architectures
_INTERMEDIATE_SIZE_ALIASES = (
    "intermediate_size",
    "ffn_dim",
    "n_inner",
    "ffn_hidden_size",
)
_MOE_INTERMEDIATE_SIZE_ALIASES = (
    "moe_intermediate_size",
    "expert_intermediate_size",
    "ffn_dim",
)


def _infer_padded_dimensions(
    padded_layers: dict[str, dict[str, list[int]]],
) -> dict[str, int]:
    """
    Infer which model config dimensions need to be updated based on padded layers.

    Analyzes the padded layer names and their new shapes to determine the correct
    padded dimension values for config fields like intermediate_size.

    Safe-to-pad layers are MLP layers where padding with zeros doesn't affect
    model correctness:
    - gate_proj/up_proj/w1/w3/fc1: out_features = intermediate_size
      (extra output channels produce zeros since there's no input contribution)
    - down_proj/w2/fc2: in_features = intermediate_size
      (extra input channels receive zeros, contributing nothing to output)

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

        # Check if this is an MoE layer
        is_moe = any(pattern in module_name for pattern in _MOE_PATTERNS)

        # Check if this is a gate/up projection (out_features = intermediate_size)
        is_gate_up = any(pattern in module_name for pattern in _MLP_GATE_UP_PATTERNS)

        # Check if this is a down projection (in_features = intermediate_size)
        is_down = any(pattern in module_name for pattern in _MLP_DOWN_PATTERNS)

        if is_gate_up:
            intermediate_size = padded_out
        elif is_down:
            intermediate_size = padded_in
        else:
            # Not a safe-to-pad layer, skip
            continue

        # Determine the config key based on whether this is MoE or regular MLP
        config_key = (
            "moe_intermediate_size" if is_moe else "intermediate_size"
        )

        # Use the largest padded value if multiple layers are padded
        if config_key not in updates or intermediate_size > updates[config_key]:
            updates[config_key] = intermediate_size

    return updates


def _find_config_key(config_data: dict, aliases: tuple[str, ...]) -> str | None:
    """Find which alias exists in the config data."""
    for alias in aliases:
        if alias in config_data:
            return alias
    return None


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
                # Find the actual config key (handles different model architectures)
                if key == "intermediate_size":
                    actual_key = _find_config_key(config_data, _INTERMEDIATE_SIZE_ALIASES)
                elif key == "moe_intermediate_size":
                    actual_key = _find_config_key(
                        config_data, _MOE_INTERMEDIATE_SIZE_ALIASES
                    )
                else:
                    actual_key = key if key in config_data else None

                if actual_key is not None:
                    logger.info(
                        f"Updating {actual_key} from {config_data[actual_key]} to {value} "
                        "due to block quantization padding"
                    )
                    config_data[actual_key] = value

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
