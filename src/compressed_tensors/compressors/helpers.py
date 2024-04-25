# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

import torch
from compressed_tensors.base import SPARSITY_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionConfig, CompressionFormat
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoConfig


__all__ = [
    "infer_compressor_from_model_config",
    "load_compressed",
    "save_compressed",
    "save_compressed_model",
]


def infer_compressor_from_model_config(
    pretrained_model_name_or_path: str,
) -> Optional[ModelCompressor]:
    """
    Given a path to a model config, extract a sparsity config if it exists and return
    the associated ModelCompressor

    :param pretrained_model_name_or_path: path to model config on disk or HF hub
    :return: matching compressor if config contains a sparsity config
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    if sparsity_config is None:
        return None

    format = sparsity_config.get("format")
    sparsity_config = CompressionConfig.load_from_registry(format, **sparsity_config)
    compressor = ModelCompressor.load_from_registry(format, config=sparsity_config)
    return compressor


def save_compressed(
    tensors: Dict[str, Tensor],
    save_path: Union[str, Path],
    compression_format: Optional[CompressionFormat] = None,
):
    """
    Save compressed tensors to disk. If tensors are not compressed,
    save them as is.

    :param tensors: dictionary of tensors to compress
    :param save_path: path to save compressed tensors
    :param compression_format: compression format used for the tensors
    :return: compression config, if tensors were compressed - None otherwise
    """
    if tensors is None or len(tensors) == 0:
        raise ValueError("No tensors or empty tensors provided to compress")

    # if no compression_format specified, default to `dense_sparsity`
    compression_format = compression_format or CompressionFormat.dense_sparsity.value

    if not (
        compression_format in ModelCompressor.registered_names()
        or compression_format in ModelCompressor.registered_aliases()
    ):
        raise ValueError(
            f"Unknown compression format: {compression_format}. "
            f"Must be one of {set(ModelCompressor.registered_names() + ModelCompressor.registered_aliases())}"  # noqa E501
        )

    # compress
    compressor = ModelCompressor.load_from_registry(compression_format)
    # save compressed tensors
    compressed_tensors = compressor.compress(tensors)
    save_file(compressed_tensors, save_path)


def load_compressed(
    compressed_tensors: Union[str, Path],
    compression_config: CompressionConfig = None,
    device: Optional[str] = "cpu",
) -> Generator[Tuple[str, Tensor], None, None]:
    """
    Load compressed tensors from disk.
    If tensors are not compressed, load them as is.

    :param compressed_tensors: path to compressed tensors.
        This can be a path to a file or a directory containing
        one or multiple safetensor files (if multiple - in the format
        assumed by huggingface)
    :param compression_config: compression config to use for decompressing tensors.
    :param device: device to move tensors to. If None, tensors are loaded on CPU.
    :param return_dict: if True, return a dictionary of decompressed tensors
    :return a generator that yields the name and tensor of the decompressed tensor
    """
    if compressed_tensors is None or not Path(compressed_tensors).exists():
        raise ValueError("No compressed tensors provided to load")

    if (
        compression_config is None
        or compression_config.format == CompressionFormat.dense_sparsity.value
    ):
        # if no compression_config specified, or `dense_sparsity` format specified,
        # assume tensors are not compressed on disk
        weight_mappings = get_weight_mappings(compressed_tensors)
        for weight_name, file_with_weight_name in weight_mappings.items():
            with safe_open(file_with_weight_name, framework="pt", device=device) as f:
                weight = f.get_tensor(weight_name)
                yield weight_name, weight
    else:
        # decompress tensors
        compression_format = compression_config.format
        compressor = ModelCompressor.load_from_registry(
            compression_format, config=compression_config
        )
        yield from compressor.decompress(compressed_tensors, device=device)


def save_compressed_model(
    model: torch.nn.Module,
    filename: str,
    compression_format: Optional[CompressionFormat] = None,
    force_contiguous: bool = True,
):
    """
    Wrapper around safetensors `save_model` helper function, which allows for
    saving compressed model to disk.

    Note: The model is assumed to have a
        state_dict with  unique entries

    :param model: model to save on disk
    :param filename: filename location to save the file
    :param compression_format: compression format used for the model
    :param force_contiguous: forcing the state_dict to be saved as contiguous tensors
    """
    state_dict = model.state_dict()
    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    try:
        save_compressed(state_dict, filename, compression_format=compression_format)
    except ValueError as e:
        msg = str(e)
        msg += " Or use save_compressed_model(..., force_contiguous=True), read the docs for potential caveats."  # noqa E501
        raise ValueError(msg)
