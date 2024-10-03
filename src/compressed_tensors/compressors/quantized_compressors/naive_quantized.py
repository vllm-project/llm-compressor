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

from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import can_quantize
from torch import Tensor


__all__ = [
    "NaiveQuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]


@BaseCompressor.register(name=CompressionFormat.naive_quantized.value)
class NaiveQuantizationCompressor(BaseQuantizationCompressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    COMPRESSION_PARAM_NAMES = [
        "weight",
        "weight_scale",
        "weight_zero_point",
        "weight_g_idx",
    ]

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
        dtype = quantization_args.pytorch_dtype()
        return {"weight": (weight_shape, dtype)}

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        zero_point: Optional[Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
        quantization_args: Optional[QuantizationArgs] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses a single uncompressed weight

        :param weight: uncompressed weight tensor
        :param scale: quantization scale for weight
        :param zero_point: quantization zero point for weight
        :param g_idx: optional mapping from column index to group index
        :param quantization_args: quantization parameters for weight
        :param device: optional device to move compressed output to
        :return: dictionary of compressed weight data
        """
        if can_quantize(weight, quantization_args):
            quantized_weight = quantize(
                x=weight,
                scale=scale,
                zero_point=zero_point,
                g_idx=g_idx,
                args=quantization_args,
                dtype=quantization_args.pytorch_dtype(),
            )

            if device is not None:
                quantized_weight = quantized_weight.to(device)

        return {"weight": quantized_weight}

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:
        """
        Decompresses a single compressed weight

        :param compressed_data: dictionary of data needed for decompression
        :param quantization_args: quantization parameters for the weight
        :return: tensor of the decompressed weight
        """
        weight = compressed_data["weight"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data.get("weight_zero_point", None)
        g_idx = compressed_data.get("weight_g_idx", None)
        decompressed_weight = dequantize(
            x_q=weight, scale=scale, zero_point=zero_point, g_idx=g_idx
        )

        return decompressed_weight


@BaseCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for integer quantized models
    """

    pass


@BaseCompressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for fp quantized models
    """

    pass
