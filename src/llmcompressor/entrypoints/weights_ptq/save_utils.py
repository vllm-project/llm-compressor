import json
import os

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
from transformers.file_utils import CONFIG_NAME

__all__ = ["update_config", "update_safetensors_index"]


def update_config(
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
):
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
    config_file_path = _find_config_path(save_directory)
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data

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
    file_path = _find_safetensors_index_path(save_directory)
    if file_path is None:
        return

    with open(file_path, "w") as file:
        json.dump(
            {
                "total_size": total_size,
                "weight_map": weight_map,
            },
            file,
            indent=2,
            sort_keys=True,
        )


def _find_config_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)

    return None


def _find_safetensors_index_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name.endswith("safetensors.index.json"):
            return os.path.join(save_directory, file_name)

    return None
