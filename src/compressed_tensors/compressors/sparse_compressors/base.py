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
from typing import TYPE_CHECKING, Dict, Generator, Optional, Set, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.utils import (
    get_nested_mappings_from_state_dict,
    get_nested_weight_mappings,
    merge_names,
)
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationScheme


__all__ = ["BaseSparseCompressor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class BaseSparseCompressor(BaseCompressor):
    """
    Base class representing a sparse compression algorithm. Each child class should
    implement compression_param_names, compress_weight and decompress_weight;

    Compressors support compressing/decompressing a full module state dict or a single
    quantized PyTorch leaf module.

    Model Load Lifecycle (run_compressed=False):
        - ModelCompressor.decompress()
            - apply_quantization_config()
            - BaseSparseCompressor.decompress()
                - BaseSparseCompressor.decompress_weight()

    Model Save Lifecycle:
        - ModelCompressor.compress()
            - BaseSparseCompressor.compress()
                - BaseSparseCompressor.compress_weight()

    Module Lifecycle (run_compressed=True):
        - apply_quantization_config()
        - compressed_module = CompressedLinear(module)
            - initialize_module_for_quantization()
            - BaseSparseCompressor.compression_param_info()
            - register_parameters()
        - compressed_module.forward()
            - compressed_module.decompress()


    :param config: config specifying compression parameters
    """

    def compress(
        self,
        model_state: Dict[str, Tensor],
        compression_targets: Optional[Set[str]] = None,
        show_progress: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict using bitmask compression

        :param model_state: state dict of uncompressed model
        :param compression_targets: optional set of layer prefixes to compress,
            otherwise compress all layers (for backwards compatibility)
        :return: compressed state dict
        """
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )
        for name, value in tqdm(
            model_state.items(),
            desc="Compressing with sparsity",
            disable=(not show_progress),
        ):
            if not self.should_compress(name, compression_targets):
                compressed_dict[name] = value
                continue
            prefix = name
            if prefix.endswith(".weight"):
                prefix = prefix[: -(len(".weight"))]

            compression_data = self.compress_weight(prefix, value)
            for key in compression_data.keys():
                if key in compressed_dict:
                    _LOGGER.warn(
                        f"Expected all compressed state_dict keys to be unique, but "
                        f"found an existing entry for {key}. The existing entry will "
                        "be replaced."
                    )

            compressed_dict.update(compression_data)

        return compressed_dict

    def decompress(
        self,
        path_to_model_or_tensors: str,
        device: str = "cpu",
        params_to_skip_load: Optional[Tuple] = None,
        **kwargs,
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a bitmask compressed state dict located
        at path_to_model_or_tensors and returns a generator
        for sequentially decompressing back to a dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: device to load decompressed weights onto
        :param params_to_skip_load: a list of non-sparsity parameters (e.g quantization
            parameters) that we want to skip loading. As the sparsity compresssor does
            not handle quantized decompression, this should contain any quantization
            parameters when decompressing stacked compressors. We want these parameters
            to be handled by the quantization decompressor
        :return: iterator for generating decompressed weights
        """
        weight_mappings, ignored_params = get_nested_weight_mappings(
            path_to_model_or_tensors,
            self.compression_param_names,
            return_unmatched_params=True,
        )
        for module_path in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[module_path].items():
                full_name = merge_names(module_path, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)

            decompressed = self.decompress_weight(weight_data)
            yield merge_names(module_path, "weight"), decompressed

        for ignored_param_name, safe_path in ignored_params.items():
            should_skip = False
            if params_to_skip_load is not None:
                for param_to_skip in params_to_skip_load:
                    if param_to_skip in ignored_param_name:
                        should_skip = True

            if not should_skip:
                with safe_open(safe_path, framework="pt", device=device) as f:
                    value = f.get_tensor(ignored_param_name)
                yield ignored_param_name, value

    def decompress_from_state_dict(
        self,
        state_dict: Dict[str, Tensor],
    ) -> Generator[Tuple[str, Dict[str, Tensor]], None, None]:
        """
        Decompress the state dict of a module (or model)

        Unlike `self.decompress`, this function does not need to explicitly skip params
        via params_to_skip_load because it is more convenient for its only caller
        (ModelCompressor.decompress_model) to retrieve all unused param keys

        :param state_dict: state dict containing parameters to decompress
        :return: Generator of (param_path, param_val)
        """
        weight_mappings, ignored_params = get_nested_mappings_from_state_dict(
            state_dict, self.compression_param_names, return_unmatched_params=True
        )

        for module_path in weight_mappings.keys():
            weight_data = {}
            for param_name, param_value in weight_mappings[module_path].items():
                weight_data[param_name] = param_value

            decompressed = self.decompress_weight(weight_data)
            yield merge_names(module_path, "weight"), decompressed

        for ignored_param_path, ignored_param_value in ignored_params.items():
            yield ignored_param_path, ignored_param_value

    @staticmethod
    def should_compress(name: str, expanded_targets: Optional[Set[str]] = None) -> bool:
        """
        Check if a parameter should be compressed.
        Currently, this only returns True for weight parameters.

        :param name: name of the parameter
        :param expanded_targets: set of layer prefixes to compress
        :return: whether or not the parameter should be compressed
        """
        if expanded_targets is None:
            return name.endswith(".weight")

        return (
            name.endswith(".weight") and name[: -(len(".weight"))] in expanded_targets
        )

    def decompress_module_from_state_dict(
        self,
        prefix: str,
        state_dict: Dict[str, torch.Tensor],
        scheme: "QuantizationScheme",
    ) -> Dict[str, torch.Tensor]:
        """
        This function is implemented as a workaround because of how
        `ModelCompressor.quantization_compressor` can be set to either
        an instance of `BaseQuantizationCompressor` or `BaseSparseCompressor`.
        """
        return state_dict.copy()
