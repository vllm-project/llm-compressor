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
from typing import Iterable, Optional

from sparsetensors.quantization.lifecycle.calibration import set_module_for_calibration
from sparsetensors.quantization.lifecycle.frozen import freeze_module_quantization
from sparsetensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from sparsetensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from sparsetensors.quantization.utils import iter_named_leaf_modules
from torch.nn import Module


__all__ = [
    "apply_quantization_config",
    "apply_quantization_status",
]


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
    for name, submodule in _iter_named_leaf_modules(model):
        if _find_first_name_or_class_match(name, submodule, config.ignore):
            continue  # layer matches ignore list, continue
        target = _find_first_name_or_class_match(name, submodule, target_to_scheme)
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
    if status >= QuantizationStatus.INITIALIZED:
        model.apply(initialize_module_for_quantization)
    if status >= QuantizationStatus.CALIBRATION:
        model.apply(set_module_for_calibration)
    if status >= QuantizationStatus.FROZEN:
        model.apply(freeze_module_quantization)


def _find_first_name_or_class_match(
    name: str,
    module: Module,
    targets: Iterable[str],
) -> Optional[str]:
    # first element of targets that matches the given name
    # if no name matches returns first target that matches the class name
    # returns None otherwise
    return _find_first_match(name, targets) or _find_first_match(
        module.__class__.__name__, targets
    )


def _find_first_match(value: str, targets: Iterable[str]) -> Optional[str]:
    # returns first element of target that matches value either
    # exactly or as a regex after 're:'
    for target in targets:
        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                return target
        elif target == value:
            return target
    return None
