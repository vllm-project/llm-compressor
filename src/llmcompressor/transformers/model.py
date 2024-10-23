import logging
from typing import Type

from compressed_tensors import COMPRESSION_CONFIG_NAME, QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from transformers import AutoConfig, PreTrainedModel


def load_model_decompressed(
    model_cls: Type[PreTrainedModel],
    pretrained_model_name_or_path: str,
    *args,
    **kwargs,
) -> PreTrainedModel:
    # temporarily set the log level to error, to ignore printing out long missing
    # and unexpected key error messages (these are EXPECTED for quantized models)
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()
    transformers_logger.setLevel(level=logging.ERROR)

    # Work around ModelCompressor.from_pretrained
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None) or getattr(
        config, COMPRESSION_CONFIG_NAME, None
    )
    compressor = ModelCompressor.from_compression_config(compression_config)

    model = model_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    if compressor is not None:
        compressor.decompress(model_path=pretrained_model_name_or_path, model=model)

    # restore transformers logging level now that model shell is loaded
    transformers_logger.setLevel(level=restore_log_level)

    return model
