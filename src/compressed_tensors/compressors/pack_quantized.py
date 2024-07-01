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

import logging
import math
from typing import Dict, Generator, Tuple

import numpy as np
import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import can_quantize
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


__all__ = ["PackedQuantizationCompressor", "pack_to_int32", "unpack_from_int32"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(Compressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int32
    """

    COMPRESSION_PARAM_NAMES = [
        "weight_packed",
        "weight_scale",
        "weight_zero_point",
        "weight_shape",
    ]

    def compress(
        self,
        model_state: Dict[str, Tensor],
        names_to_scheme: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param names_to_scheme: quantization args for each quantized weight, needed for
        quantize function to calculate bit depth
        :return: compressed state dict
        """
        compressed_dict = {}
        weight_suffix = ".weight"
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(weight_suffix):
                prefix = name[: -(len(weight_suffix))]
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                shape = torch.tensor(value.shape)
                if scale is not None and zp is not None:
                    # weight is quantized, compress it
                    quant_args = names_to_scheme[prefix]
                    if can_quantize(value, quant_args):
                        # convert weight to an int if not already compressed
                        value = quantize(
                            x=value,
                            scale=scale,
                            zero_point=zp,
                            args=quant_args,
                            dtype=torch.int8,
                        )
                    value = pack_to_int32(value.cpu(), quant_args.num_bits)
                    compressed_dict[merge_names(prefix, "weight_shape")] = shape
                    compressed_dict[merge_names(prefix, "weight_packed")] = value
                    continue

            elif name.endswith("zero_point"):
                if torch.all(value == 0):
                    # all zero_points are 0, no need to include in
                    # compressed state_dict
                    continue

            compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self,
        path_to_model_or_tensors: str,
        names_to_scheme: Dict[str, QuantizationArgs],
        device: str = "cpu",
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at path_to_model_or_tensors
        and returns a generator for sequentially decompressing back to a
        dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: optional device to load intermediate weights into
        :return: compressed state dict
        """
        weight_mappings = get_nested_weight_mappings(
            path_to_model_or_tensors, self.COMPRESSION_PARAM_NAMES
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                weight_data["num_bits"] = names_to_scheme.get(weight_name).num_bits
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)

            if "weight_scale" in weight_data:
                zero_point = weight_data.get("weight_zero_point", None)
                scale = weight_data["weight_scale"]
                weight = weight_data["weight_packed"]
                num_bits = weight_data["num_bits"]
                original_shape = torch.Size(weight_data["weight_shape"])
                unpacked = unpack_from_int32(weight, num_bits, original_shape)
                decompressed = dequantize(
                    x_q=unpacked,
                    scale=scale,
                    zero_point=zero_point,
                )
                yield merge_names(weight_name, "weight"), decompressed


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

    # convert packed input to unsigned numpy
    value = value.numpy().view(np.uint32)
    pack_factor = 32 // num_bits

    # unpack
    mask = pow(2, num_bits) - 1
    unpacked = np.zeros((value.shape[0], value.shape[1] * pack_factor))
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

    # remove padding
    original_row_size = int(shape[1])
    unpacked = unpacked[:, :original_row_size]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked.astype(np.int16) - offset).astype(np.int8)

    return torch.from_numpy(unpacked)
