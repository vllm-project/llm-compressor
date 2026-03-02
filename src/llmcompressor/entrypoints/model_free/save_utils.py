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
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert_checkpoint import Converter
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from loguru import logger

from .helpers import find_config_path, find_safetensors_index_path

__all__ = ["update_config", "update_safetensors_index"]


def update_config(
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
    converter: Converter | None = None,
):
    config_file_path = find_config_path(save_directory)

    qconfig = create_quant_config(
        config_file_path, scheme_name, scheme, ignore, converter
    )

    # construct compression (quantization) config
    qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME])
    qconfig_data = {
        COMPRESSION_VERSION_NAME: ct_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
        SPARSITY_CONFIG_NAME: {},
        TRANSFORM_CONFIG_NAME: {},
        **qconfig_data,
    }

    # write results to config.json file
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data

        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)

    else:
        logger.warning(
            f"Could not find config file in {save_directory}. Please set "
            "quantization_config to: \n"
            f"{json.dumps(qconfig_data, indent=2, sort_keys=True)}"
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


def create_quant_config(
    config_file_path: str | None,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
    converter: Converter | None = None,
) -> QuantizationConfig:
    """
    Create quantization_config in 3 possible ways:
    1) If converting from a format that isn't compressed-tensors,
        create new quant config based on converter and append scheme
    2) If checkpoint is in a pre-existing compressed-tensors format,
        use its quantization_config as starting point and append scheme
    3) Otherwise, create from scratch based on scheme
    """

    if converter is not None:
        # original checkpoint is not in compressed-tensors format
        # assume quantization_config needs be created from scratch
        qconfig = converter.create_config()
    elif config_file_path is not None:
        # load up quantization_config, if in compressed-tensors format
        # append to it instead of creating from scratch
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

            if hasattr(config_data, QUANTIZATION_CONFIG_NAME):
                qconfig = QuantizationConfig.model_validate_json(
                    config_data[QUANTIZATION_CONFIG_NAME]
                )

    if qconfig is None:
        # construct quantization config from scratch
        qconfig = QuantizationConfig.model_validate(
            {
                "config_groups": {scheme_name: scheme},
                "ignore": ignore,
                "quantization_status": QuantizationStatus.COMPRESSED,
                "format": scheme.format,
            }
        )
    else:
        scheme_name = (
            f"config_group_{len(qconfig.config_groups)}"
            if scheme_name in qconfig.config_groups
            else scheme_name
        )
        qconfig.config_groups[scheme_name] = scheme
        unique_formats = set(scheme.format for scheme in qconfig.config_groups.values())
        qconfig.format = (
            (
                next(iter(unique_formats))
                if len(unique_formats) == 1
                else CompressionFormat.mixed_precision.value
            ),
        )
        qconfig.quantization_status = QuantizationStatus.COMPRESSED

    return qconfig
