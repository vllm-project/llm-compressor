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
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Union

import torch
from compressed_tensors.config import CompressionFormat
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
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    KV_CACHE_TARGETS,
    infer_quantization_status,
    is_kv_cache_quant_scheme,
    iter_named_leaf_modules,
)
from compressed_tensors.utils.helpers import fix_fsdp_module_name, replace_module
from compressed_tensors.utils.offload import update_parameter_data
from compressed_tensors.utils.safetensors_load import get_safetensors_folder
from torch.nn import Module


__all__ = [
    "load_pretrained_quantization",
    "apply_quantization_config",
    "apply_quantization_status",
    "find_name_or_class_matches",
]

from compressed_tensors.quantization.utils.helpers import is_module_quantized
from compressed_tensors.utils.safetensors_load import get_quantization_state_dict


_LOGGER = logging.getLogger(__name__)


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


def apply_quantization_config(
    model: Module, config: QuantizationConfig, run_compressed: bool = False
) -> Dict:
    """
    Initializes the model for quantization in-place based on the given config

    :param model: model to apply quantization config to
    :param config: quantization config
    :param run_compressed: Whether the model will be run in compressed mode or
        decompressed fully on load
    """
    # remove reference to the original `config`
    # argument. This function can mutate it, and we'd
    # like to keep the original `config` as it is.
    config = deepcopy(config)
    # build mapping of targets to schemes for easier matching
    # use ordered dict to preserve target ordering in config
    target_to_scheme = OrderedDict()
    config = process_quantization_config(config)
    names_to_scheme = OrderedDict()
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    if run_compressed:
        from compressed_tensors.linear.compressed_linear import CompressedLinear

    # list of submodules to ignore
    ignored_submodules = defaultdict(list)
    # mark appropriate layers for quantization by setting their quantization schemes
    for name, submodule in iter_named_leaf_modules(model):
        # potentially fix module name to remove FSDP wrapper prefix
        name = fix_fsdp_module_name(name)
        if matches := find_name_or_class_matches(name, submodule, config.ignore):
            for match in matches:
                ignored_submodules[match].append(name)
            continue  # layer matches ignore list, continue
        targets = find_name_or_class_matches(name, submodule, target_to_scheme)
        if targets:
            scheme = _scheme_from_targets(target_to_scheme, targets, name)
            if run_compressed:
                format = config.format
                if format != CompressionFormat.dense.value:
                    if isinstance(submodule, torch.nn.Linear):
                        # TODO: expand to more module types
                        compressed_linear = CompressedLinear.from_linear(
                            submodule,
                            quantization_scheme=scheme,
                            quantization_format=format,
                        )
                        replace_module(model, name, compressed_linear)

            # target matched - add layer and scheme to target list
            submodule.quantization_scheme = _scheme_from_targets(
                target_to_scheme, targets, name
            )

            names_to_scheme[name] = submodule.quantization_scheme.weights

    if config.ignore is not None and ignored_submodules is not None:
        if set(config.ignore) - set(ignored_submodules):
            _LOGGER.warning(
                "Some layers that were to be ignored were "
                "not found in the model: "
                f"{set(config.ignore) - set(ignored_submodules)}"
            )

    # apply current quantization status across all targeted layers
    apply_quantization_status(model, config.quantization_status)
    return names_to_scheme


def process_quantization_config(config: QuantizationConfig) -> QuantizationConfig:
    """
    Preprocess the raw QuantizationConfig

    :param config: the raw QuantizationConfig
    :return: the processed QuantizationConfig
    """
    if config.kv_cache_scheme is not None:
        config = process_kv_cache_config(config)

    return config


def process_kv_cache_config(
    config: QuantizationConfig, targets: Union[List[str], str] = KV_CACHE_TARGETS
) -> QuantizationConfig:
    """
    Reformulate the `config.kv_cache` as a `config_group`
    and add it to the set of existing `config.groups`

    :param config: the QuantizationConfig
    :return: the QuantizationConfig with additional "kv_cache" group
    """
    kv_cache_dict = config.kv_cache_scheme.model_dump()
    kv_cache_scheme = QuantizationScheme(
        output_activations=QuantizationArgs(**kv_cache_dict),
        targets=targets,
    )
    kv_cache_group = dict(kv_cache=kv_cache_scheme)
    config.config_groups.update(kv_cache_group)
    return config


def apply_quantization_status(model: Module, status: QuantizationStatus):
    """
    Applies in place the quantization lifecycle up to the given status

    :param model: model to apply quantization to
    :param status: status to update the module to
    """
    current_status = infer_quantization_status(model)

    if status >= QuantizationStatus.INITIALIZED > current_status:
        force_zero_point_init = status != QuantizationStatus.COMPRESSED
        model.apply(
            lambda module: initialize_module_for_quantization(
                module, force_zero_point=force_zero_point_init
            )
        )

    if current_status < status >= QuantizationStatus.CALIBRATION > current_status:
        # only quantize weights up front when our end goal state is calibration,
        # weight quantization parameters are already loaded for frozen/compressed
        quantize_weights_upfront = status == QuantizationStatus.CALIBRATION
        model.apply(
            lambda module: set_module_for_calibration(
                module, quantize_weights_upfront=quantize_weights_upfront
            )
        )
    if current_status < status >= QuantizationStatus.FROZEN > current_status:
        model.apply(freeze_module_quantization)

    if current_status < status >= QuantizationStatus.COMPRESSED > current_status:
        model.apply(compress_quantized_weights)


def find_name_or_class_matches(
    name: str, module: Module, targets: Iterable[str], check_contains: bool = False
) -> List[str]:
    """
    Returns all targets that match the given name or the class name.
    Returns empty list otherwise.
    The order of the output `matches` list matters.
    The entries are sorted in the following order:
        1. matches on exact strings
        2. matches on regex patterns
        3. matches on module names
    """
    targets = sorted(targets, key=lambda x: ("re:" in x, x))
    if isinstance(targets, Iterable):
        matches = _find_matches(name, targets) + _find_matches(
            module.__class__.__name__, targets, check_contains
        )
        matches = [match for match in matches if match is not None]
        return matches


def _find_matches(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> List[str]:
    # returns all the targets that match value either
    # exactly or as a regex after 're:'. if check_contains is set to True,
    # additionally checks if the target string is contained with value.
    matches = []
    for target in targets:
        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                matches.append(target)
        elif check_contains:
            if target.lower() in value.lower():
                matches.append(target)
        elif target == value:
            matches.append(target)
    return matches


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
    g_idx_name = f"{base_name}_g_idx"

    state_dict_scale = state_dict.get(f"{module_name}.{scale_name}", None)
    state_dict_zp = state_dict.get(f"{module_name}.{zp_name}", None)
    state_dict_g_idx = state_dict.get(f"{module_name}.{g_idx_name}", None)

    if state_dict_scale is not None:
        # module is quantized
        update_parameter_data(module, state_dict_scale, scale_name)
        if state_dict_zp is None:
            # fill in zero point for symmetric quantization
            state_dict_zp = torch.zeros_like(state_dict_scale, device="cpu")
        update_parameter_data(module, state_dict_zp, zp_name)

    if state_dict_g_idx is not None:
        update_parameter_data(module, state_dict_g_idx, g_idx_name)


def _scheme_from_targets(
    target_to_scheme: OrderedDictType[str, QuantizationScheme],
    targets: List[str],
    name: str,
) -> QuantizationScheme:
    if len(targets) == 1:
        # if `targets` iterable contains a single element
        # use it as the key
        return target_to_scheme[targets[0]]

    # otherwise, we need to merge QuantizationSchemes corresponding
    # to multiple targets. This is most likely because `name` module
    # is being target both as an ordinary quantization target, as well
    # as kv cache quantization target
    schemes_to_merge = [target_to_scheme[target] for target in targets]
    return _merge_schemes(schemes_to_merge, name)


def _merge_schemes(
    schemes_to_merge: List[QuantizationScheme], name: str
) -> QuantizationScheme:

    kv_cache_quantization_scheme = [
        scheme for scheme in schemes_to_merge if is_kv_cache_quant_scheme(scheme)
    ]
    if not kv_cache_quantization_scheme:
        # if the schemes_to_merge do not contain any
        # kv cache QuantizationScheme
        # return the first scheme (the prioritized one,
        # since the order of schemes_to_merge matters)
        return schemes_to_merge[0]
    else:
        # fetch the kv cache QuantizationScheme and the highest
        # priority non-kv cache QuantizationScheme and merge them
        kv_cache_quantization_scheme = kv_cache_quantization_scheme[0]
        quantization_scheme = [
            scheme
            for scheme in schemes_to_merge
            if not is_kv_cache_quant_scheme(scheme)
        ][0]
        schemes_to_merge = [kv_cache_quantization_scheme, quantization_scheme]
        merged_scheme = {}
        for scheme in schemes_to_merge:
            scheme_dict = {
                k: v for k, v in scheme.model_dump().items() if v is not None
            }
            # when merging multiple schemes, the final target will be
            # the `name` argument - hence erase the original targets
            del scheme_dict["targets"]
            # make sure that schemes do not "clash" with each other
            overlapping_keys = set(merged_scheme.keys()) & set(scheme_dict.keys())
            if overlapping_keys:
                raise ValueError(
                    f"The module: {name} is being modified by two clashing "
                    f"quantization schemes, that jointly try to override "
                    f"properties: {overlapping_keys}. Fix the quantization config "
                    "so that it is not ambiguous."
                )
            merged_scheme.update(scheme_dict)

        merged_scheme.update(targets=[name])

        return QuantizationScheme(**merged_scheme)
