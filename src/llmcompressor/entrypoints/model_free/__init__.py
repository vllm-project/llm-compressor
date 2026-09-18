import os
from typing import Iterable, Optional

import torch
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.entrypoints.convert.convert_checkpoint import convert_checkpoint
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils.safetensors_load import get_checkpoint_files

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.validate import (
    validate_config,
    validate_safetensors_index,
)

__all__ = ["model_free_ptq"]


def model_free_ptq(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str | None = None,
    config: QuantizationConfig | None = None,
    ignore: Iterable[str] = tuple(),
    max_workers: int = 1,
    device: Optional[str | torch.device | list[str | torch.device]] = None,
    converter: Converter | None = None,
):
    """
    Quantize a model without the need for a model definition. This function
    operates on a model stub or folder containing weights saved in safetensors
    files.

    For microscale schemes (NVFP4, MXFP4), fused weight sets (q/k/v, gate/up)
    are handled correctly even when split across shards. Each shard job receives
    a precomputed inverse_weight_map specifying exactly which tensors to load
    from which files — enabling true partial reads with no runtime discovery
    and no redundant tensor reads.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: directory to save quantized weights to
    :param scheme: weight quantization scheme or preset scheme name.
        Mutually exclusive with config.
    :param config: quantization config containing one or more schemes and
        optional kv cache quantization. Mutually exclusive with scheme.
    :param ignore: modules to ignore. Modules ending with "norm" are
        automatically ignored
    :param max_workers: maximum number of concurrent worker threads.
        Effective concurrency may be lower when GPU memory is tight.
    :param device: device(s) for quantization. Accepts a single device
        string/object or a list. When multiple devices are given, shards
        are dynamically assigned based on real-time GPU memory.
    :param converter: optional converter to apply to the checkpoint before
        running model-free PTQ, e.g. an AWQ or fp8 dequantizer
    """
    model_files = get_checkpoint_files(model_stub)
    config = validate_config(config, scheme, ignore)
    validate_safetensors_index(model_files, config)

    mfptq = ModelFreePtqConverter(config)
    converters = ([converter] if converter is not None else []) + [mfptq]

    convert_checkpoint(
        model_stub=model_stub,
        save_directory=save_directory,
        converter=converters,
        max_workers=max_workers,
        device=device,
    )
