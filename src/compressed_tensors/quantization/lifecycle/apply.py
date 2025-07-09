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
from typing import Set, Union

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.lifecycle.compressed import (
    compress_quantized_weights,
)
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
)
from compressed_tensors.utils.helpers import fix_fsdp_module_name, replace_module
from compressed_tensors.utils.offload import update_parameter_data
from compressed_tensors.utils.safetensors_load import get_safetensors_folder
from safetensors import safe_open
from torch.nn import Module


__all__ = [
    "load_pretrained_quantization_parameters",
    "apply_quantization_config",
    "apply_quantization_status",
    "find_name_or_class_matches",
    "expand_target_names",
    "is_target",
]

from compressed_tensors.quantization.utils.helpers import is_module_quantized
from compressed_tensors.utils.safetensors_load import (
    get_quantization_parameter_to_path_mapping,
)


_LOGGER = logging.getLogger(__name__)


def load_pretrained_quantization_parameters(
    model: Module,
    model_name_or_path: Optional[str] = None,
    load_weight_quantization: Optional[bool] = False,
):
    """
    Loads the quantization parameters (scale and zero point) from model_name_or_path to
    a model that has already been initialized with a quantization config.

    NOTE: Will always load inputs/output parameters.
    Will conditioanlly load weight parameters, if load_weight_quantization is set to True.

    :param model: model to load pretrained quantization parameters to
    :param model_name_or_path: Hugging Face stub or local folder containing a quantized
        model, which is used to load quantization parameters
    :param load_weight_quantization: whether or not the weight quantization parameters shoud
        be laoded
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

        if load_weight_quantization and submodule.quantization_scheme.weights:
            base_name = "weight"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )


def apply_quantization_config(
    model: Module, config: Union[QuantizationConfig, None], run_compressed: bool = False
) -> Dict[str, QuantizationScheme]:
    """
    Initializes the model for quantization in-place based on the given config.
    Optionally coverts quantizable modules to compressed_linear modules

    :param model: model to apply quantization config to
    :param config: quantization config
    :param run_compressed: Whether the model will be run in compressed mode or
        decompressed fully on load
    """
    # Workaround for when HF Quantizer passes None, see PR #180
    if config is None:
        return dict()

    # remove reference to the original `config`
    # argument. This function can mutate it, and we'd
    # like to keep the original `config` as it is.
    config = deepcopy(config)
    # build mapping of targets to schemes for easier matching
    # use ordered dict to preserve target ordering in config
    target_to_scheme = OrderedDict()
    config = process_quantization_config(config)
    names_to_scheme = dict()
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    if run_compressed:
        from compressed_tensors.linear.compressed_linear import CompressedLinear

    # list of submodules to ignore
    ignored_submodules = defaultdict(list)
    # mark appropriate layers for quantization by setting their quantization schemes
    for name, submodule in model.named_modules():
        # potentially fix module name to remove FSDP wrapper prefix
        name = fix_fsdp_module_name(name)
        if matches := find_name_or_class_matches(name, submodule, config.ignore):
            for match in matches:
                ignored_submodules[match].append(name)
            continue  # layer matches ignore list, continue

        targets = find_name_or_class_matches(name, submodule, target_to_scheme)

        if targets:
            # mark modules to be quantized by adding
            # quant scheme to the matching layers
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
            submodule.quantization_scheme = scheme

            names_to_scheme[name] = submodule.quantization_scheme

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
    if targets == KV_CACHE_TARGETS:
        _LOGGER.info(f"KV cache targets set to default value of: {KV_CACHE_TARGETS}")

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

        # When decompressing, we set the scale_dtype as the model's dtype
        # This is because the normal workflow of using the weight's dtype
        # will be incorrect as the model weight will be compressed
        # Therfore, use the dtype set by the user using the PretrainedModel
        scale_dtype = None
        if status == QuantizationStatus.FROZEN:
            if hasattr(model, "dtype"):
                scale_dtype = model.dtype

        model.apply(
            lambda module: initialize_module_for_quantization(
                module, force_zero_point=force_zero_point_init, scale_dtype=scale_dtype
            )
        )

    if current_status < status >= QuantizationStatus.COMPRESSED > current_status:
        model.apply(compress_quantized_weights)


def expand_target_names(
    model: Module,
    targets: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> Set[str]:
    """
    Finds all unique module names in the model that match the given
    targets and ignore lists.

    Note: Targets must be regexes, layer types, or full layer names.

    :param model: model to search for targets in
    :param targets: Iterable of targets to search for
    :param ignore: Iterable of targets to ignore
    :return: set of all targets that match the given targets and should
        not be ignored
    """
    return {
        name
        for name, module in model.named_modules()
        if is_target(name, module, targets, ignore)
    }


def is_target(
    name: str,
    module: Module,
    targets: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> bool:
    """
    Determines if a module should be included in the targets based on the
    targets and ignore lists.

    Note: Targets must be regexes, layer types, or full layer names.

    :param name: name of the module
    :param module: the module itself
    :param targets: Iterable of targets to search for
    :param ignore: Iterable of targets to ignore
    :return: True if the module is a target and not ignored, False otherwise
    """
    return bool(
        find_name_or_class_matches(name, module, targets or [])
        and not find_name_or_class_matches(name, module, ignore or [])
    )


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
    from compressed_tensors import InternalModule

    if isinstance(module, InternalModule):
        return []

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
