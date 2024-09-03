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
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
import transformers
from compressed_tensors.base import (
    COMPRESSION_CONFIG_NAME,
    QUANTIZATION_CONFIG_NAME,
    SPARSITY_CONFIG_NAME,
)
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
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
from compressed_tensors.utils import get_safetensors_folder, update_parameter_data
from compressed_tensors.utils.helpers import fix_fsdp_module_name
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor", "map_modules_to_quant_args"]

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
        **kwargs,
    ) -> Optional["ModelCompressor"]:
        """
        Given a path to a model config, extract the sparsity and/or quantization
        configs and load a ModelCompressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: compressor for the extracted configs
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
        return cls.from_compression_config(compression_config)

    @classmethod
    def from_compression_config(cls, compression_config: Dict[str, Any]):
        """
        :param compression_config: compression/quantization config dictionary
            found under key "quantization_config" in HF model config
        :return: compressor for the extracted configs
        """
        if compression_config is None:
            return None

        try:
            from transformers.utils.quantization_config import CompressedTensorsConfig

            if isinstance(compression_config, CompressedTensorsConfig):
                compression_config = compression_config.to_dict()
        except ImportError:
            pass

        sparsity_config = cls.parse_sparsity_config(compression_config)
        quantization_config = cls.parse_quantization_config(compression_config)
        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None and not isinstance(
            sparsity_config, SparsityCompressionConfig
        ):
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None and not isinstance(
            quantization_config, QuantizationConfig
        ):
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

    @staticmethod
    def parse_sparsity_config(compression_config: Dict) -> Union[Dict, None]:
        if compression_config is None:
            return None
        if SPARSITY_CONFIG_NAME not in compression_config:
            return None
        if hasattr(compression_config, SPARSITY_CONFIG_NAME):
            # for loaded HFQuantizer config
            return getattr(compression_config, SPARSITY_CONFIG_NAME)
        if SPARSITY_CONFIG_NAME in compression_config:
            # for loaded HFQuantizer config from dict
            return compression_config[SPARSITY_CONFIG_NAME]

        # SparseAutoModel format
        return compression_config.get(SPARSITY_CONFIG_NAME, None)

    @staticmethod
    def parse_quantization_config(compression_config: Dict) -> Union[Dict, None]:
        if compression_config is None:
            return None

        if hasattr(compression_config, QUANTIZATION_CONFIG_NAME):
            # for loaded HFQuantizer config
            return getattr(compression_config, QUANTIZATION_CONFIG_NAME)

        if QUANTIZATION_CONFIG_NAME in compression_config:
            # for loaded HFQuantizer config from dict
            return compression_config[QUANTIZATION_CONFIG_NAME]

        # SparseAutoModel format
        quantization_config = deepcopy(compression_config)
        quantization_config.pop(SPARSITY_CONFIG_NAME, None)
        if len(quantization_config) == 0:
            quantization_config = None
        return quantization_config

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
        quantized_modules_to_args = map_modules_to_quant_args(model)
        if self.quantization_compressor is not None:
            compressed_state_dict = self.quantization_compressor.compress(
                state_dict, names_to_scheme=quantized_modules_to_args
            )
            if self.quantization_config.format != CompressionFormat.dense.value:
                self.quantization_config.quantization_status = (
                    QuantizationStatus.COMPRESSED
                )

        if self.sparsity_compressor is not None:
            compressed_state_dict = self.sparsity_compressor.compress(
                compressed_state_dict
            )

        # HACK (mgoin): Post-process step for kv cache scales to take the
        # k/v_proj module `output_scale` parameters, and store them in the
        # parent attention module as `k_scale` and `v_scale`
        #
        # Example:
        #  Replace `model.layers.0.self_attn.k_proj.output_scale`
        #  with    `model.layers.0.self_attn.k_scale`
        if (
            self.quantization_config is not None
            and self.quantization_config.kv_cache_scheme is not None
        ):
            # HACK (mgoin): We assume the quantized modules in question
            # will be k_proj and v_proj since those are the default targets.
            # We check that both of these modules have output activation
            # quantization, and additionally check that q_proj doesn't.
            q_proj_has_no_quant_output = 0
            k_proj_has_quant_output = 0
            v_proj_has_quant_output = 0
            for name, module in model.named_modules():
                if not hasattr(module, "quantization_scheme"):
                    # We still want to count non-quantized q_proj
                    if name.endswith(".q_proj"):
                        q_proj_has_no_quant_output += 1
                    continue
                out_act = module.quantization_scheme.output_activations
                if name.endswith(".q_proj") and out_act is None:
                    q_proj_has_no_quant_output += 1
                elif name.endswith(".k_proj") and out_act is not None:
                    k_proj_has_quant_output += 1
                elif name.endswith(".v_proj") and out_act is not None:
                    v_proj_has_quant_output += 1

            assert (
                q_proj_has_no_quant_output > 0
                and k_proj_has_quant_output > 0
                and v_proj_has_quant_output > 0
            )
            assert (
                q_proj_has_no_quant_output
                == k_proj_has_quant_output
                == v_proj_has_quant_output
            )

            # Move all .k/v_proj.output_scale parameters to .k/v_scale
            working_state_dict = {}
            for key in compressed_state_dict.keys():
                if key.endswith(".k_proj.output_scale"):
                    new_key = key.replace(".k_proj.output_scale", ".k_scale")
                    working_state_dict[new_key] = compressed_state_dict[key]
                elif key.endswith(".v_proj.output_scale"):
                    new_key = key.replace(".v_proj.output_scale", ".v_scale")
                    working_state_dict[new_key] = compressed_state_dict[key]
                else:
                    working_state_dict[key] = compressed_state_dict[key]
            compressed_state_dict = working_state_dict

        # HACK: Override the dtype_byte_size function in transformers to
        # support float8 types. Fix is posted upstream
        # https://github.com/huggingface/transformers/pull/30488
        transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

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
            names_to_scheme = apply_quantization_config(model, self.quantization_config)
            load_pretrained_quantization(model, model_path)
            dense_gen = self.quantization_compressor.decompress(
                model_path, names_to_scheme=names_to_scheme
            )
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
            config_data[COMPRESSION_CONFIG_NAME] = quant_config_data
        if self.sparsity_config is not None:
            sparsity_config_data = self.sparsity_config.model_dump()
            config_data[COMPRESSION_CONFIG_NAME][
                SPARSITY_CONFIG_NAME
            ] = sparsity_config_data

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def _replace_weights(self, dense_weight_generator, model):
        for name, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            split_name = name.split(".")
            prefix, param_name = ".".join(split_name[:-1]), split_name[-1]
            module = operator.attrgetter(prefix)(model)
            update_parameter_data(module, data, param_name)


def map_modules_to_quant_args(model: Module) -> Dict:
    quantized_modules_to_args = {}
    for name, submodule in iter_named_leaf_modules(model):
        if is_module_quantized(submodule):
            if submodule.quantization_scheme.weights is not None:
                name = fix_fsdp_module_name(name)
                quantized_modules_to_args[name] = submodule.quantization_scheme.weights

    return quantized_modules_to_args


# HACK: Override the dtype_byte_size function in transformers to support float8 types
# Fix is posted upstream https://github.com/huggingface/transformers/pull/30488
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8
