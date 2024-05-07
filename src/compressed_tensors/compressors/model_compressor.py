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

import json
import logging
import operator
import os
from typing import Dict, Optional, Union

from compressed_tensors.base import (
    COMPRESSION_CONFIG_NAME,
    QUANTIZATION_CONFIG_NAME,
    SPARSITY_CONFIG_NAME,
)
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
    load_pretrained_quantization,
)
from compressed_tensors.quantization.utils import (
    is_module_quantized,
    iter_named_leaf_modules,
)
from compressed_tensors.utils import get_safetensors_folder
from torch import Tensor
from torch.nn import Module, Parameter
from tqdm import tqdm
from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    Handles compression and decompression of a model with a sparsity config and/or
    quantization config.

    Compression LifeCycle
        - compressor = ModelCompressor.from_pretrained_model(model)
        - compressed_state_dict = compressor.compress(model, state_dict)
            - compressor.quantization_compressor.compress(model, state_dict)
            - compressor.sparsity_compressor.compress(model, state_dict)
        - model.save_pretrained(output_dir, state_dict=compressed_state_dict)
        - compressor.update_config(output_dir)

    Decompression LifeCycle
        - compressor = ModelCompressor.from_pretrained(comp_model_path)
        - model = AutoModel.from_pretrained(comp_model_path)
        - compressor.decompress(comp_model_path, model)
            - compressor.sparsity_compressor.decompress(comp_model_path, model)
            - compressor.quantization_compressor.decompress(comp_model_path, model)

    :param sparsity_config: config specifying sparsity compression parameters
    :param quantization_config: config specifying quantization compression parameters
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> Optional["ModelCompressor"]:
        """
        Given a path to a model config, extract the sparsity and/or quantization
        configs and load a ModelCompressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: compressor for the extracted configs
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
        if compression_config is None:
            return None

        sparsity_config = compression_config.get(SPARSITY_CONFIG_NAME, None)
        quantization_config = compression_config.get(QUANTIZATION_CONFIG_NAME, None)

        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None:
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.parse_obj(quantization_config)

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config: Union[SparsityCompressionConfig, str, None] = None,
        quantization_format: Optional[str] = None,
    ) -> Optional["ModelCompressor"]:
        """
        Given a pytorch model and optional sparsity and/or quantization configs,
        load the appropriate compressors

        :param model: pytorch model to target for compression
        :param sparsity_config: a filled in sparsity config or string corresponding
            to a sparsity compression algorithm
        :param quantization_format: string corresponding to a quantization compression
            algorithm
        :return: compressor for the extracted configs
        """
        quantization_config = QuantizationConfig.from_pretrained(
            model, format=quantization_format
        )

        if isinstance(sparsity_config, str):  # we passed in a sparsity format
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config
            )

        if sparsity_config is None and quantization_config is None:
            return None

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    def __init__(
        self,
        sparsity_config: Optional[SparsityCompressionConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.sparsity_compressor = None
        self.quantization_compressor = None

        if sparsity_config is not None:
            self.sparsity_compressor = Compressor.load_from_registry(
                sparsity_config.format, config=sparsity_config
            )
        if quantization_config is not None:
            self.quantization_compressor = Compressor.load_from_registry(
                quantization_config.format, config=quantization_config
            )

    def compress(
        self, model: Module, state_dict: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict or model with sparsity and/or quantization

        :param model: uncompressed model to compress
        :param model_state: optional uncompressed state_dict to insert into model
        :return: compressed state dict
        """
        if state_dict is None:
            state_dict = model.state_dict()

        compressed_state_dict = state_dict
        quantized_modules_to_args = _get_weight_arg_mappings(model)
        if self.quantization_compressor is not None:
            compressed_state_dict = self.quantization_compressor.compress(
                state_dict, model_quant_args=quantized_modules_to_args
            )

        if self.sparsity_compressor is not None:
            compressed_state_dict = self.sparsity_compressor.compress(
                compressed_state_dict
            )

        return compressed_state_dict

    def decompress(self, model_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into
        """
        model_path = get_safetensors_folder(model_path)
        if self.sparsity_compressor is not None:
            dense_gen = self.sparsity_compressor.decompress(model_path)
            self._replace_weights(dense_gen, model)
            setattr(model, SPARSITY_CONFIG_NAME, self.sparsity_compressor.config)

        if self.quantization_compressor is not None:
            apply_quantization_config(model, self.quantization_config)
            load_pretrained_quantization(model, model_path)
            dense_gen = self.quantization_compressor.decompress(model_path)
            self._replace_weights(dense_gen, model)

            def update_status(module):
                module.quantization_status = QuantizationStatus.FROZEN

            model.apply(update_status)
            setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs
        for sparsity and/or quantization

        :param save_directory: path to a folder containing a HF model config
        """
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if not os.path.exists(config_file_path):
            _LOGGER.warning(
                f"Could not find a valid model config file in "
                f"{save_directory}. Compression config will not be saved."
            )
            return

        with open(config_file_path, "r") as config_file:
            config_data = json.load(config_file)

        config_data[COMPRESSION_CONFIG_NAME] = {}
        if self.quantization_config is not None:
            quant_config_data = self.quantization_config.model_dump()
            config_data[COMPRESSION_CONFIG_NAME][
                QUANTIZATION_CONFIG_NAME
            ] = quant_config_data
        if self.sparsity_config is not None:
            sparsity_config_data = self.sparsity_config.model_dump()
            config_data[COMPRESSION_CONFIG_NAME][
                SPARSITY_CONFIG_NAME
            ] = sparsity_config_data

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def _replace_weights(self, dense_weight_generator, model):
        for name, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            # loading the decompressed weights into the model
            model_device = operator.attrgetter(name)(model).device
            data_new = Parameter(data.to(model_device))
            data_old = operator.attrgetter(name)(model)
            data_old.data = data_new.data


def _get_weight_arg_mappings(model: Module) -> Dict:
    quantized_modules_to_args = {}
    for name, submodule in iter_named_leaf_modules(model):
        if is_module_quantized(submodule):
            if submodule.quantization_scheme.weights is not None:
                quantized_modules_to_args[name] = submodule.quantization_scheme.weights

    return quantized_modules_to_args
