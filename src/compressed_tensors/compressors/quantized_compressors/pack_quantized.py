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
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import can_quantize
from torch import Tensor


__all__ = ["PackedQuantizationCompressor", "pack_to_int32", "unpack_from_int32"]


@BaseCompressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(BaseQuantizationCompressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int32
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return (
            "weight_packed",
            "weight_scale",
            "weight_zero_point",
            "weight_g_idx",
            "weight_shape",
        )

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
        packed_size_zp = math.ceil(weight_shape[0] / pack_factor)
        output = {
            "weight_packed": (torch.Size((weight_shape[0], packed_size)), torch.int32),
            "weight_shape": (torch.Size((2,)), torch.int32),
        }
        if not quantization_args.symmetric and quantization_args.strategy in [
            QuantizationStrategy.GROUP.value,
            QuantizationStrategy.CHANNEL.value,
        ]:
            zp_factor = (
                quantization_args.group_size
                if quantization_args.strategy == QuantizationStrategy.GROUP.value
                else weight_shape[-1]
            )

            output["weight_zero_point"] = (
                torch.Size((packed_size_zp, weight_shape[-1] // zp_factor)),
                torch.int32,
            )
        return output

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        quantization_args: QuantizationArgs,
        zero_point: Optional[Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses a single uncompressed weight

        :param weight: uncompressed weight tensor
        :param scale: quantization scale for weight
        :param quantization_args: quantization parameters for weight
        :param zero_point: quantization zero point for weight
        :param g_idx: optional mapping from column index to group index
        :param device: optional device to move compressed output to
        :return: dictionary of compressed weight data
        """
        if global_scale is not None:
            raise ValueError(
                "global_scale is not supported for the PackQuantizationCompressor"
            )

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

        # We typically don't compress zp; apart from when using the packed_compressor
        # and when storing group/channel zp
        if not quantization_args.symmetric and quantization_args.strategy in [
            QuantizationStrategy.GROUP.value,
            QuantizationStrategy.CHANNEL.value,
        ]:
            packed_zp = pack_to_int32(
                zero_point, quantization_args.num_bits, packed_dim=0
            )
            compressed_dict["weight_zero_point"] = packed_zp
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

        # NOTE: this will fail decompression as we don't currently handle packed zp on
        # decompression
        if not quantization_args.symmetric and quantization_args.strategy in [
            QuantizationStrategy.GROUP.value,
            QuantizationStrategy.CHANNEL.value,
        ]:
            raise ValueError(
                "Decompression of packed zero points is currently not supported"
            )
            assert zero_point is not None
            original_zp_shape = (original_shape[0], scale.shape[-1])
            zero_point = unpack_from_int32(
                zero_point, num_bits, original_zp_shape, packed_dim=0
            )

        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, zero_point=zero_point, g_idx=g_idx
        )

        return decompressed_weight


def pack_to_int32(
    value: torch.Tensor,
    num_bits: int,
    packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Packs a tensor of quantized weights stored in int8 into int32s with padding

    Pseudocode:
     1. Shift wrt num_bits to convert to unsigned. num_bits=8
        [1,2] -> [129, 130]
     2. Pad to fill in 32 bits
        [129, 130] -> [129, 130, 0, 0]
     3. convert to binary align in order
        [129, 130, 0, 0] -> 00000000 00000000 10000010 10000001
     4. convert aligned binary to number
        00000000000000001000001010000001 -> 33409
     5. covert back to uint32
        33409 -> 33409

    :param value: tensor to pack
    :param num_bits: number of bits used to store underlying data, must be at least 1
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original bit range.

    Return tensors in int8

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
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
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
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[0])
        unpacked = unpacked[:original_row_size, :]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked
