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
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.safetensors_load import find_config_path
from loguru import logger
from pydantic import ValidationError

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter

__all__ = ["update_config"]


def update_config(
    save_directory: str | os.PathLike,
    config: QuantizationConfig,
    converter: Converter | None = None,
) -> None:
    """
    Write the final quantization config to config.json in save_directory.

    Reads any pre-existing CT quantization_config from the checkpoint so that
    sequential ptq runs append rather than overwrite. Chains through the user
    converter (if any) then ModelFreePtqConverter to produce the final config.
    """
    config_file_path = find_config_path(save_directory)

    # Read config.json once; seed incoming from any pre-existing CT quant config
    config_data: dict | None = None
    incoming: QuantizationConfig | None = None
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)
        if QUANTIZATION_CONFIG_NAME in config_data:
            qdata = config_data[QUANTIZATION_CONFIG_NAME]
            qdata.pop(COMPRESSION_VERSION_NAME, None)
            try:
                incoming = QuantizationConfig.model_validate(qdata)
            except ValidationError:
                incoming = None

    # Chain: user converter (if any) then mfptq
    if converter is not None and hasattr(converter, "update_config"):
        incoming = converter.update_config(incoming)
    mfptq = ModelFreePtqConverter(config)
    final_config = mfptq.update_config(incoming)

    qconfig_data = final_config.model_dump()
    qconfig_data = {
        COMPRESSION_VERSION_NAME: ct_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
        TRANSFORM_CONFIG_NAME: {},
        **qconfig_data,
    }

    if config_data is not None:
        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data
        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)
    else:
        logger.warning(
            f"Could not find config file in {save_directory}. Please set "
            "quantization_config to: \n"
            f"{json.dumps(qconfig_data, indent=2, sort_keys=True)}"
        )
