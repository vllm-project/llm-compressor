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

from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Tuple, Union

import torch
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs, QuantizationConfig
from compressed_tensors.registry import RegistryMixin
from torch import Tensor
from torch.nn import Module


__all__ = ["BaseCompressor"]


class BaseCompressor(RegistryMixin, ABC):
    """
    Base class representing a model compression algorithm. Each child class should
    implement compression_param_info, compress_weight and decompress_weight.

    Compressors support compressing/decompressing a full module state dict or a single
    quantized PyTorch leaf module.

    Model Load Lifecycle (run_compressed=False):
        - ModelCompressor.decompress()
            - apply_quantization_config()
            - BaseCompressor.decompress()

    Model Save Lifecycle:
        - ModelCompressor.compress()
            - BaseCompressor.compress()


    Module Lifecycle (run_compressed=True):
        - apply_quantization_config()
        - compressed_module = CompressedLinear(module)
            - initialize_module_for_quantization()
            - BaseCompressor.compression_param_info()
            - register_parameters()
        - compressed_module.forward()
            -compressed_module.decompress()


    :param config: config specifying compression parameters
    """

    def __init__(
        self, config: Union[SparsityCompressionConfig, QuantizationConfig, None] = None
    ):
        self.config = config

    def compression_param_info(
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
        """
        Creates a dictionary of expected shapes and dtypes for each compression
            parameter used by the compressor

        :param weight_shape: uncompressed weight shape
        :param quantization_args: quantization parameters for the weight
        :return: dictionary mapping compressed parameter names to shape and dtype
        """
        raise NotImplementedError()

    @abstractmethod
    def compress(
        self,
        model_state: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param kwargs: additional arguments for compression
        :return: compressed state dict
        """
        raise NotImplementedError()

    @abstractmethod
    def decompress(
        self,
        path_to_model_or_tensors: str,
        device: str = "cpu",
        **kwargs,
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at path_to_model_or_tensors
        and returns a generator for sequentially decompressing back to a
        dense state dict

        :param path_to_model_or_tensors: path to compressed safetensors model (directory
            with one or more safetensors files) or compressed tensors file
        :param names_to_scheme: quantization args for each quantized weight
        :param device: optional device to load intermediate weights into
        :return: compressed state dict
        """
        raise NotImplementedError()

    def compress_module(self, module: Module) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compresses a single quantized leaf PyTorch module. If the module is not
        quantized, this function has no effect.

        :param module: PyTorch module to compress
        :return: dictionary of compressed weight data, or None if module is not
            quantized
        """
        if not hasattr(module, "quantization_scheme"):
            return None  # module is not quantized
        quantization_scheme = module.quantization_scheme
        if not hasattr(quantization_scheme, "weights"):
            return None  # weights are not quantized

        quantization_args = quantization_scheme.weights
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        weight_zero_point = getattr(module, "weight_zero_point", None)

        return self.compress_weight(
            weight=weight,
            scale=weight_scale,
            zero_point=weight_zero_point,
            quantization_args=quantization_args,
        )

    def compress_weight(
        self,
        weight: Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses a single uncompressed weight

        :param weight: uncompressed weight tensor
        :param kwargs: additional arguments for compression
        """
        raise NotImplementedError()

    def decompress_module(self, module: Module):
        """
        Decompresses a single compressed leaf PyTorch module. If the module is not
        quantized, this function has no effect.

        :param module: PyTorch module to decompress
        :return: tensor of the decompressed weight, or None if module is not quantized
        """
        if not hasattr(module, "quantization_scheme"):
            return None  # module is not quantized
        quantization_scheme = module.quantization_scheme
        if not hasattr(quantization_scheme, "weights"):
            return None  # weights are not quantized

        quantization_args = quantization_scheme.weights
        compressed_data = {}
        for name, parameter in module.named_parameters():
            compressed_data[name] = parameter

        return self.decompress_weight(
            compressed_data=compressed_data, quantization_args=quantization_args
        )

    def decompress_weight(
        self, compressed_data: Dict[str, Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Decompresses a single compressed weight

        :param compressed_data: dictionary of data needed for decompression
        :param kwargs: additional arguments for decompression
        :return: tensor of the decompressed weight
        """
        raise NotImplementedError()
