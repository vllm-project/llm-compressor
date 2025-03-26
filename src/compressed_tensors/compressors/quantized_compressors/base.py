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
from pathlib import Path
from typing import Any, Dict, Generator, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.utils import (
    get_nested_mappings_from_state_dict,
    get_nested_weight_mappings,
    merge_names,
)
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)

__all__ = ["BaseQuantizationCompressor"]


class BaseQuantizationCompressor(BaseCompressor):
    """
    Base class representing a quant compression algorithm. Each child class should
    implement compression_param_info, compress_weight and decompress_weight.

    Compressors support compressing/decompressing a full module state dict or a single
    quantized PyTorch leaf module.

    Model Load Lifecycle (run_compressed=False):
        - ModelCompressor.decompress()
            - apply_quantization_config()
            - BaseQuantizationCompressor.decompress()
                - BaseQuantizationCompressor.decompress_weight()

    Model Save Lifecycle:
        - ModelCompressor.compress()
            - BaseQuantizationCompressor.compress()
                - BaseQuantizationCompressor.compress_weight()

    Module Lifecycle (run_compressed=True):
        - apply_quantization_config()
        - compressed_module = CompressedLinear(module)
            - initialize_module_for_quantization()
            - BaseQuantizationCompressor.compression_param_info()
            - register_parameters()
        - compressed_module.forward()
            - compressed_module.decompress()


    :param config: config specifying compression parameters
    """

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
        input_zp_suffix = ".input_zero_point"
        weight_zp_suffix = ".weight_zero_point"
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Quantized Compression"):
            # check if the parameter we're compressing is the weight zp
            # or the input zp
            is_weight_zp = name.endswith(weight_zp_suffix)
            is_input_zp = name.endswith(input_zp_suffix)

            # if we're saving the weight zp, fetch weight quant args
            if is_weight_zp:
                quant_args_zp = names_to_scheme.get(name[: -(len(weight_zp_suffix))])
                if isinstance(quant_args_zp, tuple):
                    # If tuple, first value is weight args, second is input args
                    quant_args_zp = quant_args_zp[0]

            # if we're saving the input zp, fetch input quant args
            if is_input_zp:
                input_args_zp = names_to_scheme.get(name[: -(len(input_zp_suffix))])
                if isinstance(input_args_zp, tuple):
                    # If tuple, first value is weight args, second is input args
                    input_args_zp = input_args_zp[-1]

            if name.endswith(weight_suffix):
                prefix = name[: -(len(weight_suffix))]
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                g_idx = model_state.get(merge_names(prefix, "weight_g_idx"), None)
                if scale is not None:
                    # weight is quantized, compress it
                    if isinstance(names_to_scheme[prefix], tuple):
                        quant_args = names_to_scheme[prefix][0]
                    else:
                        quant_args = names_to_scheme[prefix]

                    compressed_data = self.compress_weight(
                        weight=value,
                        scale=scale,
                        zero_point=zp,
                        g_idx=g_idx,
                        quantization_args=quant_args,
                        device="cpu",
                    )
                    for key, value in compressed_data.items():
                        compressed_dict[merge_names(prefix, key)] = value
                else:
                    compressed_dict[name] = value.to("cpu")
            # only save if asym
            elif is_weight_zp and quant_args_zp.symmetric:
                continue
            # only save if asym
            elif is_input_zp and input_args_zp.symmetric:
                continue
            elif name.endswith("g_idx") and torch.any(value <= -1):
                continue
            else:
                compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self,
        path_to_model_or_tensors: Union[str, Path, Dict[str, Any]],
        names_to_scheme: Dict[str, QuantizationArgs],
        device: str = "cpu",
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
        if isinstance(path_to_model_or_tensors, (str, Path)):
            yield from self._decompress_from_path(
                path_to_model_or_tensors, names_to_scheme, device
            )

        else:
            yield from self._decompress_from_state_dict(
                path_to_model_or_tensors, names_to_scheme
            )

    def _decompress_from_path(self, path_to_model, names_to_scheme, device):
        weight_mappings = get_nested_weight_mappings(
            path_to_model, self.compression_param_names
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)
            if "weight_scale" in weight_data:
                quant_args = names_to_scheme[weight_name]
                decompressed = self.decompress_weight(
                    compressed_data=weight_data, quantization_args=quant_args
                )
                yield merge_names(weight_name, "weight"), decompressed

    def _decompress_from_state_dict(self, state_dict, names_to_scheme):
        weight_mappings = get_nested_mappings_from_state_dict(
            state_dict, self.compression_param_names
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, param_value in weight_mappings[weight_name].items():
                weight_data[param_name] = param_value

            if "weight_scale" in weight_data:
                quant_args = names_to_scheme[weight_name]
                decompressed = self.decompress_weight(
                    compressed_data=weight_data, quantization_args=quant_args
                )
                yield merge_names(weight_name, "weight"), decompressed
