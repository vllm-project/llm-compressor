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

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Union

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.modeling import (
    initialize_hooked_attention,
    initialize_hooked_kv_cache,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
    is_attention_module,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.utils.helpers import replace_module
from compressed_tensors.utils.match import (
    is_narrow_match,
    match_named_modules,
    match_targets,
)
from compressed_tensors.utils.offload import update_parameter_data
from compressed_tensors.utils.safetensors_load import get_safetensors_folder
from loguru import logger
from safetensors import safe_open
from torch.nn import Module


__all__ = [
    "load_pretrained_quantization_parameters",
    "apply_quantization_config",
]

from compressed_tensors.quantization.utils.helpers import is_module_quantized
from compressed_tensors.utils.safetensors_load import (
    get_quantization_parameter_to_path_mapping,
)


def load_pretrained_quantization_parameters(
    model: Module,
    model_name_or_path: Optional[str] = None,
    load_weight_qparams: Optional[bool] = False,
):
    """
    Loads the quantization parameters (scale and zero point) from model_name_or_path to
    a model that has already been initialized with a quantization config.

    NOTE: Will always load inputs/output parameters. Will conditioanlly load weight
    parameters, if load_weight_qparams is set to True.

    :param model: model to load pretrained quantization parameters to
    :param model_name_or_path: Hugging Face stub or local folder containing a quantized
        model, which is used to load quantization parameters
    :param load_weight_qparams: whether or not the weight quantization parameters
        should be loaded
    """
    model_path = get_safetensors_folder(model_name_or_path)
    mapping = get_quantization_parameter_to_path_mapping(model_path)

    for name, submodule in model.named_modules():
        if not is_module_quantized(submodule):
            continue
        if submodule.quantization_scheme.input_activations is not None:
            base_name = "input"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )
        if submodule.quantization_scheme.output_activations is not None:
            base_name = "output"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )

        if load_weight_qparams and submodule.quantization_scheme.weights:
            base_name = "weight"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )


def apply_quantization_config(
    model: Module, config: Union[QuantizationConfig, None], run_compressed: bool = False
):
    """
    Initializes the model for quantization in-place based on the given config.
    Optionally coverts quantizable modules to compressed_linear modules

    :param model: model to apply quantization config to
    :param config: quantization config
    :param run_compressed: Whether the model will be run in compressed mode or
        decompressed fully on load
    """
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    config = deepcopy(config)
    if config is None:  # see PR #180
        return dict()

    # force zero points during initialization
    force_zero_point = config.quantization_status != QuantizationStatus.COMPRESSED

    # apply and initialize kv cache quantization
    if config.kv_cache_scheme is not None:
        _apply_kv_cache_scheme(
            model, config.kv_cache_scheme, config.quantization_status
        )

    # build mapping of targets to schemes for easier matching
    # use ordered dict to preserve target ordering in config
    target_to_scheme = OrderedDict()
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    # mark appropriate layers for quantization by setting their quantization schemes
    for name, submodule in match_named_modules(
        model, target_to_scheme, config.ignore, warn_on_fail=True
    ):
        # mark modules to be quantized by adding
        # quant scheme to the matching layers
        matched_targets = match_targets(name, submodule, target_to_scheme)
        scheme = _scheme_from_targets(target_to_scheme, matched_targets, name)
        # target matched - add layer and scheme to target list
        submodule.quantization_scheme = scheme

        # replace with run compressed if applicable
        # FUTURE: move this to model compressor
        if (
            run_compressed
            and isinstance(submodule, torch.nn.Linear)
            and config.format != CompressionFormat.dense.value
        ):
            # TODO: expand to more module types
            compressed_linear = CompressedLinear.from_linear(
                submodule,
                quantization_scheme=scheme,
                quantization_format=config.format,
            )
            replace_module(model, name, compressed_linear)

        else:
            if is_attention_module(submodule) and is_narrow_match(
                model, scheme.targets, name
            ):
                initialize_hooked_attention(model, submodule)

            initialize_module_for_quantization(
                submodule,
                force_zero_point=force_zero_point,
            )

        submodule.quantization_status = config.quantization_status


def _apply_kv_cache_scheme(
    model: torch.nn.Module,
    kv_cache_scheme: QuantizationArgs,
    status: QuantizationStatus,
):
    if not kv_cache_scheme.symmetric:
        raise logger.warning("vLLM does not support asymmetric kv cache quantization")

    # applies and initializes kv cache quantization
    # this step cannot come after attention apply/initialize
    # otherwise it will override the attention qparams
    scheme = QuantizationScheme(
        targets=[".*self_attn$"],  # is never read in practice
        input_activations=kv_cache_scheme,
    )
    for submodule in model.modules():
        if is_attention_module(submodule):
            submodule.quantization_scheme = scheme
            initialize_hooked_kv_cache(model, submodule)
            initialize_module_for_quantization(submodule, force_zero_point=False)
            submodule.quantization_status = status


def _load_quant_args_from_mapping(
    base_name: str, module_name: str, module: Module, mapping: Dict
):
    # TODO: skip update and just register here, don't do it in initialize
    """
    Loads scale and zero point from a state_dict into the specified module

    :param base_name: quantization target, one of: weights, input_activations or
    output_activations
    :param module_name: pytorch module name to look up in state_dict
    :module: pytorch module associated with module_name
    :mapping: mapping to search fetch paths on disk for a given parameter
    """
    scale_name = f"{base_name}_scale"
    zp_name = f"{base_name}_zero_point"
    g_idx_name = f"{base_name}_g_idx"

    state_dict_scale_path = mapping.get(f"{module_name}.{scale_name}", None)
    state_dict_zp_path = mapping.get(f"{module_name}.{zp_name}", None)
    state_dict_g_idx_path = mapping.get(f"{module_name}.{g_idx_name}", None)

    if state_dict_g_idx_path is not None:
        with safe_open(state_dict_g_idx_path, framework="pt", device="cpu") as f:
            state_dict_g_idx = f.get_tensor(f"{module_name}.{g_idx_name}")

        update_parameter_data(module, state_dict_g_idx, g_idx_name)

    if state_dict_scale_path is not None:
        # module is quantized
        with safe_open(state_dict_scale_path, framework="pt", device="cpu") as f:
            state_dict_scale = f.get_tensor(f"{module_name}.{scale_name}")

        update_parameter_data(module, state_dict_scale, scale_name)

        if state_dict_zp_path is None:
            # fill in zero point for symmetric quantization
            state_dict_zp = torch.zeros_like(state_dict_scale, device="cpu")
        else:
            with safe_open(state_dict_zp_path, framework="pt", device="cpu") as f:
                state_dict_zp = f.get_tensor(f"{module_name}.{zp_name}")

        update_parameter_data(module, state_dict_zp, zp_name)


def _scheme_from_targets(
    target_to_scheme: OrderedDictType[str, QuantizationScheme],
    targets: List[str],
    name: str,
) -> QuantizationScheme:
    # return the first scheme (the prioritized one,
    # since the order of target_to_scheme matters)
    return target_to_scheme[targets[0]]
