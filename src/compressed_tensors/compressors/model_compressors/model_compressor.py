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
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, TypeVar, Union

import compressed_tensors
import torch
import transformers
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.sparse_compressors import DenseCompressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    load_pretrained_quantization_parameters,
)
from compressed_tensors.transform import TransformConfig
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    get_execution_device,
    get_offloaded_device,
    get_safetensors_folder,
    has_offloaded_params,
    register_offload_parameter,
    update_parameter_data,
)
from compressed_tensors.utils.helpers import (
    fix_fsdp_module_name,
    is_compressed_tensors_config,
)
from compressed_tensors.utils.match import match_named_modules
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME


if TYPE_CHECKING:
    from compressed_tensors.compressors import BaseQuantizationCompressor


__all__ = ["ModelCompressor", "map_module_to_scheme"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # dummy type if not available from transformers
    CompressedTensorsConfig = TypeVar("CompressedTensorsConfig")


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

    sparsity_config: Optional[SparsityCompressionConfig] = None
    quantization_config: Optional[QuantizationConfig] = None
    transform_config: Optional[TransformConfig] = None

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
        :return: compressor for the configs, or None if model is not compressed
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        return cls.from_compression_config(compression_config)

    @classmethod
    def from_compression_config(
        cls,
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ):
        """
        :param compression_config:
            A compression or quantization config

            The type is one of the following:
            1. A Dict found under either "quantization_config" or "compression_config"
                keys in the config.json
            2. A CompressedTensorsConfig found under key "quantization_config" in HF
                model config
        :return: compressor for the configs, or None if model is not compressed
        """
        if compression_config is None:
            return None

        sparsity_config = cls.parse_sparsity_config(compression_config)
        quantization_config = cls.parse_quantization_config(compression_config)
        # TODO: transform config is not support by CompressedTensorsConfig yet

        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None:
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.model_validate(quantization_config)

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config: Union[SparsityCompressionConfig, str, None] = None,
        quantization_format: Optional[Union[str, List[str]]] = None,
    ) -> Optional["ModelCompressor"]:
        """
        Given a pytorch model and optional sparsity and/or quantization configs,
        load the appropriate compressors

        :param model: pytorch model to target for compression
        :param sparsity_config: a filled in sparsity config or string corresponding
            to a sparsity compression algorithm
        :param quantization_format: string corresponding to a quantization compression
            algorithm
        :return: compressor for the configs, or None if model is not compressed
        """
        quantization_config = QuantizationConfig.from_pretrained(
            model, format=quantization_format
        )

        # use config passed as argument
        if isinstance(sparsity_config, str):  # we passed in a sparsity format
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config
            )

        # use config attached to model
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        if not any((quantization_config, sparsity_config, transform_config)):
            return None

        return cls(
            sparsity_config=sparsity_config,
            quantization_config=quantization_config,
            transform_config=transform_config,
            compression_formats=[quantization_format]
            if isinstance(quantization_format, str)
            else quantization_format,
        )

    @staticmethod
    def parse_sparsity_config(
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ) -> Union[Dict[str, Any], None]:
        """
        Parse sparsity config from quantization/compression config. Sparsity
        config is nested inside q/c config

        :param compression_config: quantization/compression config
        :return: sparsity config
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            s_config = compression_config.sparsity_config
            return s_config.model_dump() if s_config is not None else None

        # explicitly return None if {} in config
        return compression_config.get(SPARSITY_CONFIG_NAME, None) or None

    @staticmethod
    def parse_quantization_config(
        compression_config: Union[Dict[str, Any], "CompressedTensorsConfig"],
    ) -> Union[Dict[str, Any], None]:
        """
        Parse quantization config from quantization/compression config. The
        quantization are all the fields that are not the sparsity config or
        metadata fields

        :param compression_config: quantization/compression config
        :return: quantization config without sparsity config or metadata fields
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            q_config = compression_config.quantization_config
            return q_config.model_dump() if q_config is not None else None

        quantization_config = deepcopy(compression_config)
        quantization_config.pop(SPARSITY_CONFIG_NAME, None)
        quantization_config.pop(TRANSFORM_CONFIG_NAME, None)

        # some fields are required, even if a qconfig is not present
        # pop them off and if nothing remains, then there is no qconfig
        quant_method = quantization_config.pop(QUANTIZATION_METHOD_NAME, None)
        _ = quantization_config.pop(COMPRESSION_VERSION_NAME, None)

        if len(quantization_config) == 0:
            return None

        # replace popped off values
        # note that version is discarded for now
        if quant_method is not None:
            quantization_config[QUANTIZATION_METHOD_NAME] = quant_method

        return quantization_config

    def _fetch_unique_quantization_formats(self) -> List[str]:
        """
        Get all unique compression formats present in a model.
        :return: list of quantization formats
        """
        quantization_formats = []
        for _, scheme in self.quantization_config.config_groups.items():
            if scheme.format is not None and scheme.format not in quantization_formats:
                quantization_formats.append(scheme.format)

        if (
            len(quantization_formats) == 0
            and self.quantization_config.format
            != CompressionFormat.mixed_precision.value
        ):
            quantization_formats.append(self.quantization_config.format)
        return quantization_formats

    def __init__(
        self,
        sparsity_config: Optional[SparsityCompressionConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        transform_config: Optional[TransformConfig] = None,
        compression_formats: Optional[List[str]] = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.transform_config = transform_config
        self.compression_formats = compression_formats

        self.sparsity_compressor = None
        self.quantization_compressor: Optional[
            Dict[str, Union[BaseQuantizationCompressor, DenseCompressor]]
        ] = None
        # no transform compressor is required

        if sparsity_config is not None:
            self.sparsity_compressor = BaseCompressor.load_from_registry(
                sparsity_config.format, config=sparsity_config
            )

        if quantization_config is not None:
            # If a list of compression_format is not provided, we resolve the
            # relevant quantization formats using the config groups from the config
            # and if those are not defined, we fall-back to the global quantization fmt
            if not self.compression_formats:
                self.compression_formats = self._fetch_unique_quantization_formats()

            self.quantization_compressor = {}
            for format in self.compression_formats:
                self.quantization_compressor[
                    format
                ] = BaseCompressor.load_from_registry(
                    format, config=quantization_config
                )

    # ----- model memory compression/decompression pathways ----- #

    def compress_model(self, model: Module):
        """
        Compress a model in memory. Because the model structure is modified in place,
        this method is more memory-efficient than `self.compress`

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = [
            module_name
            for module_name, _module in match_named_modules(
                model=model,
                targets=self.sparsity_config.targets if self.sparsity_config else [],
                ignore=self.sparsity_config.ignore if self.sparsity_config else [],
            )
        ]
        for prefix, module in tqdm(
            match_named_modules(
                model,
                [*sparse_compression_targets, *module_to_scheme.keys()],
                warn_on_fail=True,
            ),
            desc="Compressing model",
        ):
            module_device = get_execution_device(module)
            is_meta = module_device.type == "meta"

            exec_device = "meta" if is_meta else "cpu"
            onloading_device = "meta" if is_meta else module_device

            # in the future, support compression on same device
            with align_module_device(module, execution_device=exec_device):
                state_dict = {
                    f"{prefix}.{name}": param
                    for name, param in module.named_parameters(recurse=False)
                }

            # quantization first
            if prefix in module_to_scheme:
                if (
                    not hasattr(module.quantization_scheme, "format")
                    or module.quantization_scheme.format is None
                ):
                    if len(self.compression_formats) > 1:
                        raise ValueError(
                            "Applying multiple compressors without defining "
                            "per module formats is not supported "
                        )
                    format = self.compression_formats[0]
                else:
                    format = module.quantization_scheme.format

                quant_compressor = self.quantization_compressor.get(format)
                state_dict = quant_compressor.compress(
                    state_dict,
                    names_to_scheme=module_to_scheme,
                    show_progress=False,
                    compression_device=exec_device,
                )

            # sparsity second
            if prefix in sparse_compression_targets:
                state_dict = self.sparsity_compressor.compress(
                    state_dict,
                    compression_targets=sparse_compression_targets,
                    show_progress=False,
                )

            # remove any existing parameters
            offload_device = get_offloaded_device(module)
            for name, _ in list(module.named_parameters(recurse=False)):
                delete_offload_parameter(module, name)

            # replace with compressed parameters
            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(onloading_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                register_offload_parameter(module, name, param, offload_device)

            module.quantization_status = QuantizationStatus.COMPRESSED
        # TODO: consider sparse compression to also be compression
        if (
            self.quantization_config is not None
            and self.quantization_config.format != CompressionFormat.dense.value
        ):
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

    def decompress_model(self, model: Module):
        """
        Decompress a model in memory. Because the model structure is modified in place,
        this method does not require loading some compression parameters from disk

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = [
            module_name
            for module_name, _module in match_named_modules(
                model=model,
                targets=self.sparsity_config.targets if self.sparsity_config else [],
                ignore=self.sparsity_config.ignore if self.sparsity_config else [],
            )
        ]

        for prefix, module in tqdm(
            match_named_modules(
                model,
                [*sparse_compression_targets, *module_to_scheme.keys()],
                warn_on_fail=True,
            ),
            desc="Decompressing model",
        ):
            # in the future, support decompression on same device
            with align_module_device(module, execution_device="cpu"):
                state_dict = {
                    f"{prefix}.{name}": param
                    for name, param in module.named_parameters(recurse=False)
                }

            # sparsity first
            if prefix in sparse_compression_targets:
                # sparse_compression_targets are automatically inferred by this fn
                generator = self.sparsity_compressor.decompress_from_state_dict(
                    state_dict,
                )
                # generates (param_path, param_val)
                # of compressed and unused params
                state_dict = {key: value for key, value in generator}

            # quantization second
            if prefix in module_to_scheme:
                if (
                    not hasattr(module.quantization_scheme, "format")
                    or module.quantization_scheme.format is None
                ):
                    if len(self.compression_formats) > 1:
                        raise ValueError(
                            "Applying multiple compressors without defining "
                            "per module formats is not supported "
                        )
                    format = self.compression_formats[0]
                else:
                    format = module.quantization_scheme.format
                quant_compressor = self.quantization_compressor.get(format)
                state_dict = quant_compressor.decompress_module_from_state_dict(
                    prefix,
                    state_dict,
                    scheme=module_to_scheme[prefix],
                )

            # remove any existing parameters
            exec_device = get_execution_device(module)
            offload_device = get_offloaded_device(module)
            for name, _ in list(module.named_parameters(recurse=False)):
                delete_offload_parameter(module, name)

            # replace with decompressed parameters
            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(exec_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                register_offload_parameter(module, name, param, offload_device)

            module.quantization_status = QuantizationStatus.FROZEN

    # ----- state dict compression pathways ----- #

    def compress(
        self,
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
        show_progress: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict or model with sparsity and/or quantization

        :param model: uncompressed model to compress
        :param state_dict: optional uncompressed state_dict to insert into model
        :return: compressed state dict
        """

        if state_dict is None:
            state_dict = model.state_dict()

        if self.quantization_compressor is not None:
            module_to_scheme = map_module_to_scheme(model)
            # Note - compress only supports one compression format atm
            quant_compressor = next(iter(self.quantization_compressor.values()))
            state_dict = quant_compressor.compress(
                state_dict,
                names_to_scheme=module_to_scheme,
                show_progress=show_progress,
            )

            # TODO: consider sparse compression to also be compression
            if self.quantization_config.format != CompressionFormat.dense.value:
                self.quantization_config.quantization_status = (
                    QuantizationStatus.COMPRESSED
                )

        if self.sparsity_compressor is not None:
            sparse_compression_targets: Set[str] = {
                module_name
                for module_name, _module in match_named_modules(
                    model=model,
                    targets=self.sparsity_config.targets,
                    ignore=self.sparsity_config.ignore,
                )
            }
            state_dict = self.sparsity_compressor.compress(
                state_dict,
                compression_targets=sparse_compression_targets,
                show_progress=show_progress,
            )

        # HACK: Override the dtype_byte_size function in transformers to
        # support float8 types. Fix is posted upstream
        # https://github.com/huggingface/transformers/pull/30488
        transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

        return state_dict

    # ----- disk decompression pathways ----- #

    def decompress(self, model_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into

        Note: decompress makes use of both _replace_sparsity_weights and
        _replace_weights. The variations in these methods are a result of the subtle
        variations between the sparsity and quantization compressors. Specifically,
        quantization compressors return not just the decompressed weight, but the
        quantization parameters (e.g scales, zero_point) whereas sparsity compressors
        only return the decompressed weight.

        """
        model_path = get_safetensors_folder(model_path)
        sparse_decompressed = False
        quant_compressor = (
            next(iter(self.quantization_compressor.values()))
            if self.quantization_compressor is not None
            else None
        )

        if (
            self.sparsity_compressor is not None
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            # note - decompress only supports one compressor atm
            params_to_ignore = None
            if quant_compressor is not None:
                params_to_ignore = quant_compressor.compression_param_names
            # Sparse decompression is applied on the model_path
            # The compressor will try and load any quantization parameters as well
            # params_to_skip_load will skip over quantization params from being loaded
            dense_gen = self.sparsity_compressor.decompress(
                model_path, params_to_skip_load=params_to_ignore
            )
            self._replace_sparsity_weights(dense_gen, model)
            setattr(model, SPARSITY_CONFIG_NAME, self.sparsity_compressor.config)
            sparse_decompressed = True

        if quant_compressor is not None:
            # Temporarily set quantization status to FROZEN to prevent
            # quantization during apply_quantization_config. This ensures
            # that the dtypes of the weights are not unintentionally updated.
            # The status is restored after quantization params are loaded.

            with override_quantization_status(
                self.quantization_config, QuantizationStatus.FROZEN
            ):
                apply_quantization_config(model, self.quantization_config)
                names_to_scheme: Set[QuantizationScheme] = {
                    name: getattr(module, "quantization_scheme")
                    for name, module in model.named_modules()
                    if getattr(module, "quantization_scheme", None) is not None
                }
                # Load activation scales/zp or any other quantization parameters
                # Conditionally load the weight quantization parameters if we have a
                # dense compressor or if a sparsity compressor has already been applied
                load_weight_qparams = sparse_decompressed or isinstance(
                    quant_compressor, DenseCompressor
                )
                load_pretrained_quantization_parameters(
                    model,
                    model_path,
                    # TODO: all weight quantization params will be moved to the
                    # compressor in a follow-up including initialization
                    load_weight_qparams=load_weight_qparams,
                )
            model_path_or_state_dict = (
                model.state_dict() if sparse_decompressed else model_path
            )

            dense_gen = quant_compressor.decompress(
                model_path_or_state_dict, names_to_scheme=names_to_scheme
            )
            # TODO: all weight quantization params will be moved to the compressor
            # to prevent duplicate parameter updates in update_parameter_data
            self._replace_weights(
                dense_gen, model, load_weight_qparams=not load_weight_qparams
            )

            def freeze_quantization_status(module):
                module.quantization_status = QuantizationStatus.FROZEN

            model.apply(freeze_quantization_status)
            setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs
        for sparsity and/or quantization

        :param save_directory: path to a folder containing a HF model config
        """
        # this check is also done in `from_pretrained_model`,
        # but not in `from_pretrained`` or `from_compression_config``
        if not any(
            (self.quantization_config, self.sparsity_config, self.transform_config)
        ):
            return

        # write to config.json file, regardless of whether it exists already
        # overwrite previous config and version if already existing
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        # serialize configs into json
        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        sconfig_data = (
            self.sparsity_config.model_dump()
            if self.sparsity_config is not None
            else {}
        )
        tconfig_data = (
            self.transform_config.model_dump()
            if self.transform_config is not None
            else {}
        )

        # construct compression (quantization) config
        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: sconfig_data,
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        # write results to config.json file
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def _replace_sparsity_weights(self, dense_weight_generator, model: Module):
        """
        Replace the weights of the model with the
        provided dense weights.

        This method iterates over the dense_weight_generator and
        updates the corresponding weights in the model. If a parameter
        name does not exist in the model, it will be skipped.

        :param dense_weight_generator (generator): A generator that yields
            tuples of (name, data), where 'name' is the parameter name and
            'data' is the updated param data
        :param model: The model whose weights are to be updated.
        """
        for name, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            split_name = name.split(".")
            prefix, param_name = ".".join(split_name[:-1]), split_name[-1]
            module = operator.attrgetter(prefix)(model)

            params_device = next(module.parameters()).device
            device = "cpu" if has_offloaded_params(module) else params_device
            delattr(module, param_name)
            requires_grad = data.dtype in (torch.float16, torch.float32, torch.bfloat16)
            param = torch.nn.Parameter(data.to(device), requires_grad=requires_grad)
            register_offload_parameter(module, param_name, param)

    def _replace_weights(
        self, dense_weight_generator, model: Module, load_weight_qparams: bool = True
    ):
        """
        Replace the weights of the model with the
        provided dense weights.

        This method iterates over the dense_weight_generator and
        updates the corresponding weights in the model. If a parameter
        name does not exist in the model, it will be skipped.

        :param dense_weight_generator (generator): A generator that yields
            tuples of (name, data), where 'name' is the parameter name and
            'data' is the updated param data
        :param model: The model whose weights are to be updated.
        """

        for mod_path, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            module = operator.attrgetter(mod_path)(model)

            params_device = next(module.parameters()).device
            device = "cpu" if has_offloaded_params(module) else params_device

            for param_name, param_data in data.items():
                if hasattr(module, param_name):
                    # If compressed, will have an incorrect dtype for transformers >4.49
                    # TODO: we can also just skip initialization of scales/zp if in
                    # decompression in init to be consistent with loading which happens
                    # later as well however, update_data does a good shape check -
                    # should be moved to the compressor

                    if param_name == "weight":
                        delattr(module, param_name)
                        requires_grad = param_data.dtype in (
                            torch.float16,
                            torch.float32,
                            torch.bfloat16,
                        )
                        param = torch.nn.Parameter(
                            param_data.to(device), requires_grad=requires_grad
                        )
                        register_offload_parameter(module, param_name, param)
                    elif load_weight_qparams:
                        # Should already be registered to the correct device for
                        # for scales/zero-points
                        update_parameter_data(module, param_data, param_name)


def map_module_to_scheme(model: Module) -> Dict[str, QuantizationScheme]:
    """
    Returns a dictionary which maps quantized module names to their quantization
    schemes. Only includes modules with weight quantization
    """
    return {
        fix_fsdp_module_name(name): module.quantization_scheme
        for name, module in model.named_modules()
        if (
            hasattr(module, "quantization_scheme")
            and module.quantization_scheme.weights is not None
        )
    }


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


@contextmanager
def override_quantization_status(
    config: QuantizationConfig, status: QuantizationStatus
):
    """
    Within this context, the quantization status will be set to the
    supplied status. After the context exits, the original status
    will be restored.

    :param config: the quantization config to override
    :param status: the status to temporarily set
    """
    original_status = config.quantization_status
    config.quantization_status = status
    try:
        yield
    finally:
        config.quantization_status = original_status
