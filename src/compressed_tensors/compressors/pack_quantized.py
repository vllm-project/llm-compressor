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


__all__ = ["PackedQuantizationCompressor", "pack_4bit_ints", "unpack_4bit_ints"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(Compressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int32
    """

    COMPRESSION_PARAM_NAMES = [
        "weight",
        "weight_scale",
        "weight_zero_point",
        "weight_shape",
    ]

    def compress(
        self,
        model_state: Dict[str, Tensor],
        model_quant_args: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :param model_quant_args: quantization args for each quantized weight, needed for
        quantize function to calculate bit depth
        :return: compressed state dict
        """
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(".weight"):
                prefix = name.removesuffix(".weight")
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                shape = torch.tensor(value.shape)
                if scale is not None and zp is not None:
                    # weight is quantized, compress it
                    # weight is quantized, compress it
                    quant_args = model_quant_args[prefix]
                    if can_quantize(value, quant_args):
                        # convert weight to an int if not already compressed
                        value = quantize(
                            x=value,
                            scale=scale,
                            zero_point=zp,
                            args=quant_args,
                            dtype=torch.int8,
                        )
                        value = pack_4bit_ints(value.cpu())
                    compressed_dict[merge_names(prefix, "weight_shape")] = shape

            compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
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
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)

            if len(weight_data) == len(self.COMPRESSION_PARAM_NAMES):
                weight = weight_data["weight"]
                original_shape = torch.Size(weight_data["weight_shape"])
                unpacked = unpack_4bit_ints(weight, original_shape)
                decompressed = dequantize(
                    x_q=unpacked,
                    scale=weight_data["weight_scale"],
                    zero_point=weight_data["weight_zero_point"],
                )
                yield merge_names(weight_name, "weight"), decompressed


def pack_4bit_ints(value: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of int4 weights stored in int8 into int32s with padding

    :param value: tensor to pack
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    # need to convert to unsigned 8bit to use numpy's pack/unpack
    temp = (value - 8).to(torch.uint8)
    bits = np.unpackbits(temp.numpy(), axis=-1, bitorder="little")
    ranges = np.array([range(x, x + 4) for x in range(0, bits.shape[1], 8)]).flatten()
    only_4_bits = bits[:, ranges]  # top 4 bits are 0 because we're really uint4

    # pad each row to fill a full 32bit int
    pack_depth = 32
    padding = (
        math.ceil(only_4_bits.shape[1] / pack_depth) * pack_depth - only_4_bits.shape[1]
    )
    padded_bits = np.pad(
        only_4_bits, pad_width=[(0, 0), (0, padding)], constant_values=0
    )

    # after packbits each uint8 is two packed uint4s
    # then we keep the bit pattern the same but convert to int32
    compressed = np.packbits(padded_bits, axis=-1, bitorder="little")
    compressed = np.ascontiguousarray(compressed).view(np.int32)

    return torch.from_numpy(compressed)


def unpack_4bit_ints(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Unpacks a tensor packed int4 weights into individual int8s, maintaining the
    original their int4 range

    :param value: tensor to upack
    :param shape: shape to unpack into, used to remove padding
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    # unpack bits and undo padding to nearest int32 bits
    individual_depth = 4
    as_uint8 = value.numpy().view(np.uint8)
    bits = np.unpackbits(as_uint8, axis=-1, bitorder="little")
    original_row_size = int(shape[1] * individual_depth)
    bits = bits[:, :original_row_size]

    # reformat each packed uint4 to a uint8 by filling to top 4 bits with zeros
    # (uint8 format is required by np.packbits)
    shape_8bit = (bits.shape[0], bits.shape[1] * 2)
    bits_as_8bit = np.zeros(shape_8bit, dtype=np.uint8)
    ranges = np.array([range(x, x + 4) for x in range(0, shape_8bit[1], 8)]).flatten()
    bits_as_8bit[:, ranges] = bits

    # repack the bits to uint8
    repacked = np.packbits(bits_as_8bit, axis=-1, bitorder="little")

    # bits are packed in unsigned format, reformat to signed
    # update the value range from uint4 to int4
    final = repacked.astype(np.int8) - 8

    return torch.from_numpy(final)
