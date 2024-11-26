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
import math
from typing import Dict, Optional, Tuple

import numpy as np
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


__all__ = ["PackedQuantizationCompressor", "pack_to_int32", "unpack_from_int32"]


@BaseCompressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(BaseQuantizationCompressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int32
    """

    COMPRESSION_PARAM_NAMES = [
        "weight_packed",
        "weight_scale",
        "weight_zero_point",
        "weight_g_idx",
        "weight_shape",
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
        pack_factor = 32 // quantization_args.num_bits
        packed_size = math.ceil(weight_shape[1] / pack_factor)
        return {
            "weight_packed": (torch.Size((weight_shape[0], packed_size)), torch.int32),
            "weight_shape": (torch.Size((2,)), torch.int32),
        }

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
        compressed_dict = {}
        if can_quantize(weight, quantization_args):
            quantized_weight = quantize(
                x=weight,
                scale=scale,
                zero_point=zero_point,
                g_idx=g_idx,
                args=quantization_args,
                dtype=torch.int8,
            )
        else:
            quantized_weight = weight

        packed_weight = pack_to_int32(quantized_weight, quantization_args.num_bits)
        weight_shape = torch.tensor(weight.shape)
        if device is not None:
            packed_weight = packed_weight.to(device)
            weight_shape = weight_shape.to(device)

        compressed_dict["weight_shape"] = weight_shape
        compressed_dict["weight_packed"] = packed_weight

        return compressed_dict

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
        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data.get("weight_zero_point", None)
        g_idx = compressed_data.get("weight_g_idx", None)
        original_shape = torch.Size(compressed_data["weight_shape"])
        num_bits = quantization_args.num_bits
        unpacked = unpack_from_int32(weight, num_bits, original_shape)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, zero_point=zero_point, g_idx=g_idx
        )

        return decompressed_weight


def pack_to_int32(value: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Packs a tensor of quantized weights stored in int8 into int32s with padding

    :param value: tensor to pack
    :param num_bits: number of bits used to store underlying data
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    # convert to unsigned for packing
    offset = pow(2, num_bits) // 2
    value = (value + offset).to(torch.uint8)
    value = value.cpu().numpy().astype(np.uint32)
    pack_factor = 32 // num_bits

    # pad input tensor and initialize packed output
    packed_size = math.ceil(value.shape[1] / pack_factor)
    packed = np.zeros((value.shape[0], packed_size), dtype=np.uint32)
    padding = packed.shape[1] * pack_factor - value.shape[1]
    value = np.pad(value, pad_width=[(0, 0), (0, padding)], constant_values=0)

    # pack values
    for i in range(pack_factor):
        packed |= value[:, i::pack_factor] << num_bits * i

    # convert back to signed and torch
    packed = np.ascontiguousarray(packed).view(np.int32)
    return torch.from_numpy(packed)


def unpack_from_int32(
    value: torch.Tensor, num_bits: int, shape: torch.Size
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original their bit range

    :param value: tensor to upack
    :param num_bits: number of bits to unpack each data point into
    :param shape: shape to unpack into, used to remove padding
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if num_bits > 8:
        raise ValueError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits

    # unpack
    mask = pow(2, num_bits) - 1
    unpacked = torch.zeros(
        (value.shape[0], value.shape[1] * pack_factor),
        device=value.device,
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

    # remove padding
    original_row_size = int(shape[1])
    unpacked = unpacked[:, :original_row_size]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked
