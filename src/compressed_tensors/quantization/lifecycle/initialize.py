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
from enum import Enum
from typing import Optional

import torch
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.quant_args import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import is_kv_cache_quant_scheme
from compressed_tensors.utils import (
    disable_hf_hook,
    has_offloaded_params,
    register_offload_parameter,
)
from torch.nn import Module, Parameter


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
    "KVCacheScaleType",
]


_LOGGER = logging.getLogger(__name__)


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
    force_zero_point: bool = True,
):
    """
    attaches appropriate scales, zero points, and observers to a layer
    given its target quantization scheme

    apply to full model with `model.apply(initialize_module_for_quantization)`

    :param module: module to set for calibration
    :param scheme: scheme to use for quantization. if None is provided,
        will attempt to use scheme stored in the module under `quantization_scheme`,
        if not provided, the layer will be skipped
    :param force_zero_point: whether to force initialization of a zero point for
        symmetric quantization
    """
    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        # no scheme passed and layer not targeted for quantization - skip
        return

    if is_attention_module(module):
        # quantized actions based on calltime status
        _initialize_attn_scales(module)

    else:

        if scheme.input_activations is not None:
            _initialize_scale_zero_point(
                module,
                "input",
                scheme.input_activations,
                force_zero_point=force_zero_point,
            )
        if scheme.weights is not None:
            if hasattr(module, "weight"):
                weight_shape = None
                if isinstance(module, torch.nn.Linear):
                    weight_shape = module.weight.shape
                _initialize_scale_zero_point(
                    module,
                    "weight",
                    scheme.weights,
                    weight_shape=weight_shape,
                    force_zero_point=force_zero_point,
                )
            else:
                _LOGGER.warning(
                    f"module type {type(module)} targeted for weight quantization but "
                    "has no attribute weight, skipping weight quantization "
                    f"for {type(module)}"
                )

        if scheme.output_activations is not None:
            if not is_kv_cache_quant_scheme(scheme):
                _initialize_scale_zero_point(
                    module, "output", scheme.output_activations
                )

        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.INITIALIZED

        with disable_hf_hook(module):
            # wrap forward call of module to perform
            # quantized actions based on calltime status
            wrap_module_forward_quantized(module, scheme)


def is_attention_module(module: Module):
    return "attention" in module.__class__.__name__.lower() and (
        hasattr(module, "k_proj")
        or hasattr(module, "v_proj")
        or hasattr(module, "qkv_proj")
    )


def _initialize_scale_zero_point(
    module: Module,
    base_name: str,
    quantization_args: QuantizationArgs,
    weight_shape: Optional[torch.Size] = None,
    force_zero_point: bool = True,
):
    if quantization_args.dynamic:
        return

    # begin on the same device as other parameters or cpu if offloaded.
    # in the offloaded case, there's no point moving tensors to the execution device
    # if they're going to be immediately offloaded by `register_offload_parameter`
    params_device = next(module.parameters()).device
    device = "cpu" if has_offloaded_params(module) else params_device

    # infer expected scale/zero point shape
    if quantization_args.strategy == QuantizationStrategy.TOKEN:
        expected_shape = (1, 1)
    else:
        expected_shape = 1

    if base_name == "weight" and weight_shape is not None:
        if quantization_args.strategy == QuantizationStrategy.CHANNEL:
            # (output_channels, 1)
            expected_shape = (weight_shape[0], 1)
        elif quantization_args.strategy == QuantizationStrategy.GROUP:
            num_groups = weight_shape[1] // quantization_args.group_size
            expected_shape = (weight_shape[0], max(num_groups, 1))

    scale_dtype = module.weight.dtype
    if scale_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        scale_dtype = torch.float16

    # initializes empty scale, zero point, and g_idx parameters for the module
    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    register_offload_parameter(module, f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        zp_dtype = quantization_args.pytorch_dtype()
        init_zero_point = Parameter(
            torch.zeros(expected_shape, device=device, dtype=zp_dtype),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_zero_point", init_zero_point)

    # only grouped activation ordering has g_idx
    if quantization_args.actorder == ActivationOrdering.GROUP:
        g_idx_shape = (weight_shape[1],)
        g_idx_dtype = torch.int
        init_g_idx = Parameter(
            torch.full(g_idx_shape, -1, device=device, dtype=g_idx_dtype),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_g_idx", init_g_idx)


def _initialize_attn_scales(module: Module) -> None:
    """Initlaize k_scale, v_scale for  self_attn"""

    expected_shape = 1  # per tensor

    param = next(module.parameters())
    scale_dtype = param.dtype
    device = param.device

    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )

    module.register_parameter(KVCacheScaleType.KEY.value, init_scale)

    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    module.register_parameter(KVCacheScaleType.VALUE.value, init_scale)
