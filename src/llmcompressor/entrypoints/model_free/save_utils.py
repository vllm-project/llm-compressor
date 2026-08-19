import json
import os

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.utils.safetensors_load import find_config_path
from loguru import logger
from pydantic import ValidationError

__all__ = ["update_config"]


def update_config(
    save_directory: str | os.PathLike,
    config: QuantizationConfig,
    converter: Converter | None = None,
):
    """
    Update Quantization config for model stub in save_directory.
    Quantization config will either be created or updated, see
    create_or_update_quant_config docstring for more info.
    """
    config_file_path = find_config_path(save_directory)

    qconfig = create_or_update_quant_config(config_file_path, config, converter)

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
    config: QuantizationConfig,
    converter: Converter | None = None,
) -> QuantizationConfig:
    """
    Create or update quantization_config in 3 possible ways:
    1) If converting from a format that isn't compressed-tensors,
        create new quant config based on converter and append config groups
    2) If checkpoint is in a pre-existing compressed-tensors format,
        use its quantization_config as starting point and append config groups
    3) Otherwise, use the provided config directly
    """

    existing_config = None
    if converter is not None:
        existing_config = converter.create_config()
    elif config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        if QUANTIZATION_CONFIG_NAME in config_data:
            qconfigdata = config_data[QUANTIZATION_CONFIG_NAME]
            qconfigdata.pop(COMPRESSION_VERSION_NAME, None)
            try:
                existing_config = QuantizationConfig.model_validate(qconfigdata)
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

    # determine format from schemes
    unique_formats = {
        scheme.format
        for scheme in config.config_groups.values()
        if scheme.format is not None
    }
    if len(unique_formats) == 1:
        format_ = next(iter(unique_formats))
    elif len(unique_formats) > 1:
        format_ = "mixed-precision"
    else:
        format_ = "dense"

    new_config = QuantizationConfig(
        config_groups=config.config_groups,
        kv_cache_scheme=config.kv_cache_scheme,
        ignore=list(config.ignore) if config.ignore else [],
        quantization_status=QuantizationStatus.COMPRESSED,
        format=format_,
    )

    if existing_config is None:
        return new_config
    else:
        existing_config.merge(new_config)
        return existing_config
