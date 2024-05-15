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

import re
from collections import OrderedDict
from typing import Dict, Iterable, Optional

from compressed_tensors.quantization.lifecycle.calibration import (
    set_module_for_calibration,
)
from compressed_tensors.quantization.lifecycle.compressed import (
    compress_quantized_weights,
)
from compressed_tensors.quantization.lifecycle.frozen import freeze_module_quantization
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.utils import (
    infer_quantization_status,
    iter_named_leaf_modules,
)
from compressed_tensors.utils.safetensors_load import get_safetensors_folder
from torch.nn import Module


__all__ = [
    "load_pretrained_quantization",
    "apply_quantization_config",
    "apply_quantization_status",
    "find_first_name_or_class_match",
]

from compressed_tensors.quantization.utils.helpers import is_module_quantized
from compressed_tensors.utils.safetensors_load import get_quantization_state_dict


def load_pretrained_quantization(model: Module, model_name_or_path: str):
    """
    Loads the quantization parameters (scale and zero point) from model_name_or_path to
    a model that has already been initialized with a quantization config

    :param model: model to load pretrained quantization parameters to
    :param model_name_or_path: Hugging Face stub or local folder containing a quantized
    model, which is used to load quantization parameters
    """
    model_path = get_safetensors_folder(model_name_or_path)
    state_dict = get_quantization_state_dict(model_path)

    for name, submodule in iter_named_leaf_modules(model):
        if not is_module_quantized(submodule):
            continue
        if submodule.quantization_scheme.weights is not None:
            base_name = "weight"
            _load_quant_args_from_state_dict(
                base_name=base_name,
                module_name=name,
                module=submodule,
                state_dict=state_dict,
            )
        if submodule.quantization_scheme.input_activations is not None:
            base_name = "input"
            _load_quant_args_from_state_dict(
                base_name=base_name,
                module_name=name,
                module=submodule,
                state_dict=state_dict,
            )
        if submodule.quantization_scheme.output_activations is not None:
            base_name = "output"
            _load_quant_args_from_state_dict(
                base_name=base_name,
                module_name=name,
                module=submodule,
                state_dict=state_dict,
            )


def apply_quantization_config(model: Module, config: QuantizationConfig):
    """
    Initializes the model for quantization in-place based on the given config

    :param model: model to apply quantization config to
    :param config: quantization config
    """
    # build mapping of targets to schemes for easier matching
    # use ordered dict to preserve target ordering in config
    target_to_scheme = OrderedDict()
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    # mark appropriate layers for quantization by setting their quantization schemes
    for name, submodule in iter_named_leaf_modules(model):
        if find_first_name_or_class_match(name, submodule, config.ignore):
            continue  # layer matches ignore list, continue
        target = find_first_name_or_class_match(name, submodule, target_to_scheme)
        if target is not None:
            # target matched - add layer and scheme to target list
            submodule.quantization_scheme = target_to_scheme[target]

    # apply current quantization status across all targeted layers
    apply_quantization_status(model, config.quantization_status)


def apply_quantization_status(model: Module, status: QuantizationStatus):
    """
    Applies in place the quantization lifecycle up to the given status

    :param model: model to apply quantization to
    :param status: status to update the module to
    """
    current_status = infer_quantization_status(model)

    if status >= QuantizationStatus.INITIALIZED > current_status:
        model.apply(initialize_module_for_quantization)

    if current_status < status >= QuantizationStatus.CALIBRATION > current_status:
        model.apply(set_module_for_calibration)

    if current_status < status >= QuantizationStatus.FROZEN > current_status:
        model.apply(freeze_module_quantization)

    if current_status < status >= QuantizationStatus.COMPRESSED > current_status:
        model.apply(compress_quantized_weights)


def find_first_name_or_class_match(
    name: str, module: Module, targets: Iterable[str], check_contains: bool = False
) -> Optional[str]:
    # first element of targets that matches the given name
    # if no name matches returns first target that matches the class name
    # returns None otherwise
    return _find_first_match(name, targets) or _find_first_match(
        module.__class__.__name__, targets, check_contains
    )


def _find_first_match(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> Optional[str]:
    # returns first element of target that matches value either
    # exactly or as a regex after 're:'. if check_contains is set to True,
    # additionally checks if the target string is contained with value.
    for target in targets:
        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                return target
        elif check_contains:
            if target.lower() in value.lower():
                return target
        elif target == value:
            return target
    return None


def _infer_status(model: Module) -> Optional[QuantizationStatus]:
    for module in model.modules():
        status = getattr(module, "quantization_status", None)
        if status is not None:
            return status
    return None


def _load_quant_args_from_state_dict(
    base_name: str, module_name: str, module: Module, state_dict: Dict
):
    """
    Loads scale and zero point from a state_dict into the specified module

    :param base_name: quantization target, one of: weights, input_activations or
    output_activations
    :param module_name: pytorch module name to look up in state_dict
    :module: pytorch module associated with module_name
    :state_dict: state_dict to search for matching quantization parameters
    """
    scale_name = f"{base_name}_scale"
    zp_name = f"{base_name}_zero_point"
    device = next(module.parameters()).device

    scale = getattr(module, scale_name)
    zp = getattr(module, zp_name)
    scale.data = state_dict[f"{module_name}.{scale_name}"].to(device)
    zp.data = state_dict[f"{module_name}.{zp_name}"].to(device)
