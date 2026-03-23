import json
import os

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import Converter, find_config_path
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from loguru import logger
from pydantic import ValidationError

__all__ = ["update_config"]


def update_config(
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
    converter: Converter | None = None,
):
    """
    Update Quantization config for model stub in save_directory,
    based on the provided scheme and converter.
    Quantization config will either be created or updated, see
    create_or_update_quant_config docstring for more info.
    """
    config_file_path = find_config_path(save_directory)

    qconfig = create_or_update_quant_config(
        config_file_path, scheme_name, scheme, ignore, converter
    )

    # construct compression (quantization) config
    qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME])
    qconfig_data = {
        COMPRESSION_VERSION_NAME: ct_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
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


def create_or_update_quant_config(
    config_file_path: str | None,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
    converter: Converter | None = None,
) -> QuantizationConfig:
    """
    Create or update quantization_config in 3 possible ways:
    1) If converting from a format that isn't compressed-tensors,
        create new quant config based on converter and append scheme
    2) If checkpoint is in a pre-existing compressed-tensors format,
        use its quantization_config as starting point and append scheme
    3) Otherwise, create from scratch based on scheme
    """

    qconfig = None
    if converter is not None:
        # original checkpoint is not in compressed-tensors format
        # assume quantization_config needs be created from scratch
        qconfig = converter.create_config()
    elif config_file_path is not None:
        # load up quantization_config, if pre-existing compressed-tensors
        # format exists, append to it instead of creating from scratch
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        if QUANTIZATION_CONFIG_NAME in config_data:
            qconfigdata = config_data[QUANTIZATION_CONFIG_NAME]
            # version in json but not allowed in QuantizationConfig
            qconfigdata.pop(COMPRESSION_VERSION_NAME, None)
            try:
                qconfig = QuantizationConfig.model_validate(qconfigdata)
            except ValidationError as e:
                logger.warning(
                    "Unable to parse original checkpoint quantization_config. "
                    f"The quantization_config will be created from scratch: {e}"
                )
        else:
            logger.info(
                "No pre-existing quantization_config found. "
                "The quantization_config will be created from scratch"
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
        # update pre-existing quantization config
        scheme_name = (
            f"config_group_{len(qconfig.config_groups)}"
            if scheme_name in qconfig.config_groups
            else scheme_name
        )
        qconfig.config_groups[scheme_name] = scheme
        unique_formats = set(scheme.format for scheme in qconfig.config_groups.values())
        qconfig.format = (
            next(iter(unique_formats))
            if len(unique_formats) == 1
            else CompressionFormat.mixed_precision.value
        )
        qconfig.quantization_status = QuantizationStatus.COMPRESSED

    return qconfig
