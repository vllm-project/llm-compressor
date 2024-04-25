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
from typing import Optional

import torch
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Module, Parameter


__all__ = [
    "initialize_module_for_quantization",
]


_LOGGER = logging.getLogger(__name__)


def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
):
    """
    attaches appropriate scales, zero points, and observers to a layer
    given its target quantization scheme

    apply to full model with `model.apply(initialize_module_for_quantization)`

    :param module: module to set for calibration
    :param scheme: scheme to use for quantization. if None is provided,
        will attempt to use scheme stored in the module under `quantization_scheme`,
        if not provided, the layer will be skipped
    """
    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        # no scheme passed and layer not targeted for quantization - skip
        return

    if scheme.input_activations is not None:
        _initialize_scale_zero_point_observer(module, "input", scheme.input_activations)
    if scheme.weights is not None:
        if hasattr(module, "weight"):
            _initialize_scale_zero_point_observer(module, "weight", scheme.weights)
        else:
            _LOGGER.warning(
                f"module type {type(module)} targeted for weight quantization but "
                "has no attribute weight, skipping weight quantization "
                f"for {type(module)}"
            )
    if scheme.output_activations is not None:
        _initialize_scale_zero_point_observer(
            module, "output", scheme.output_activations
        )

    module.quantization_scheme = scheme
    module.quantization_status = QuantizationStatus.INITIALIZED

    # wrap forward call of module to perform quantized actions based on calltime status
    wrap_module_forward_quantized(module, scheme)


def _initialize_scale_zero_point_observer(
    module: Module, base_name: str, quantization_args: QuantizationArgs
):
    # initialize observer module and attach as submodule
    observer = quantization_args.get_observer()
    module.register_module(f"{base_name}_observer", observer)

    if quantization_args.dynamic:
        return  # no need to register a scale and zero point for a dynamic observer

    device = next(module.parameters()).device

    # initializes empty scale and zero point parameters for the module
    init_scale = Parameter(torch.empty(0, device=device), requires_grad=False)
    module.register_parameter(f"{base_name}_scale", init_scale)

    init_zero_point = Parameter(
        torch.empty(0, device=device, dtype=int), requires_grad=False
    )
    module.register_parameter(f"{base_name}_zero_point", init_zero_point)
