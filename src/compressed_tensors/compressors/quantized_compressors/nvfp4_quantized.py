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

import numpy
import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from torch import Tensor


__all__ = ["pack_fp4_to_uint8", "unpack_fp4_from_uint8"]

FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


@BaseCompressor.register(name=CompressionFormat.nvfp4_pack_quantized.value)
class NVFP4PackedCompressor(BaseQuantizationCompressor):
    """
    Implements compression of FP4 values. Weights of each quantized layer
    are packed into uint8. Only supports symmetric weight compression for now.
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
            "weight_global_scale",
        )

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        global_scale: Tensor,
        quantization_args: QuantizationArgs,
        device: Optional[torch.device] = None,
        zero_point: Optional[torch.Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            global_scale=global_scale,
            zero_point=zero_point,
            args=quantization_args,
        )
        compressed_dict = {}
        weight_packed = pack_fp4_to_uint8(quantized_weight)
        if device is not None:
            weight_packed = weight_packed.to(device)
        compressed_dict["weight_packed"] = weight_packed
        return compressed_dict

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:

        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        global_scale = compressed_data["weight_global_scale"]
        m, n = weight.shape
        # TODO: use a user provided dequant dtype
        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, global_scale=global_scale, dtype=unpacked.dtype
        )

        return decompressed_weight


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.
    As there are 16 valid fp4 values, two fp4 values can be
    packed into one uint8. Each fp4 value is mapped to its
    particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
    to index 7) which is then represented using 4 bits. Consecutive
    pairs of 4 bits are then packed into an uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """

    m, n = x.shape
    device = x.device

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_indices = torch.zeros_like(abs_x, dtype=torch.long)
    for i, val in enumerate(kE2M1):
        abs_indices = torch.where(torch.isclose(abs_x, val), i, abs_indices)

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x) << 3).to(torch.long)

    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)

    # Handle odd length by padding if necessary
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])

    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)

    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    return packed.reshape(m, n // 2)


kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# reference: : https://github.com/vllm-project/vllm/pull/16362
def unpack_fp4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four corresond to a consecutive
    fp4 value). The bits represent an index, which are mapped to an fp4 value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)
