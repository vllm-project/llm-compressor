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

from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


__all__ = ["BaseSparseCompressor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class BaseSparseCompressor(BaseCompressor):
    """
    Base class representing a sparse compression algorithm. Each child class should
    implement compression_param_info, compress_weight and decompress_weight.

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

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict using bitmask compression

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )
        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            compression_data = self.compress_weight(name, value)
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
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a bitmask compressed state dict located
        at path_to_model_or_tensors and returns a generator
        for sequentially decompressing back to a dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: device to load decompressed weights onto
        :return: iterator for generating decompressed weights
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
            decompressed = self.decompress_weight(weight_data)
            yield weight_name, decompressed
