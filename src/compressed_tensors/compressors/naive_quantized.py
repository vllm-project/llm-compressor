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
from typing import Dict, Generator, Tuple

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


__all__ = [
    "QuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.naive_quantized.value)
class QuantizationCompressor(Compressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    COMPRESSION_PARAM_NAMES = ["weight", "weight_scale", "weight_zero_point"]

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
                if scale is not None and zp is not None:
                    # weight is quantized, compress it
                    quant_args = names_to_scheme[prefix]
                    if can_quantize(value, quant_args):
                        # only quantize if not already quantized
                        value = quantize(
                            x=value,
                            scale=scale,
                            zero_point=zp,
                            args=quant_args,
                            dtype=quant_args.pytorch_dtype(),
                        )
            elif name.endswith("zero_point"):
                if torch.all(value == 0):
                    # all zero_points are 0, no need to include in
                    # compressed state_dict
                    continue
            compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
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

            if "weight_scale" in weight_data:
                zero_point = weight_data.get("weight_zero_point", None)
                scale = weight_data["weight_scale"]
                decompressed = dequantize(
                    x_q=weight_data["weight"],
                    scale=scale,
                    zero_point=zero_point,
                )
                yield merge_names(weight_name, "weight"), decompressed


@Compressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(QuantizationCompressor):
    """
    Alias for integer quantized models
    """

    pass


@Compressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(QuantizationCompressor):
    """
    Alias for fp quantized models
    """

    pass
