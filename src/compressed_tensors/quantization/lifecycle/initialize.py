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
import math
from enum import Enum
from typing import List, Optional

import torch
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    generate_global_scale,
    is_fp4,
    is_kv_cache_quant_scheme,
    iter_named_quantizable_modules,
)
from compressed_tensors.utils import (
    disable_hf_hook,
    get_execution_device,
    register_offload_parameter,
    update_parameter_data,
)
from torch.nn import Module, Parameter


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
    "KVCacheScaleType",
    "update_fused_layer_weight_global_scales",
]


_LOGGER = logging.getLogger(__name__)


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
    force_zero_point: bool = True,
    scale_dtype: Optional[torch.dtype] = None,
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
    :param scale_dtype: dtype to used for the scales, if overriding the
        weight dtype as the scale dtype
    """
    # TODO: don't initialize parameters when running decompression
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
                scale_dtype=scale_dtype,
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
                    scale_dtype=scale_dtype,
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
                    module, "output", scheme.output_activations, scale_dtype=scale_dtype
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
    scale_dtype: Optional[torch.dtype] = None,
):
    if quantization_args.dynamic:
        return

    # initialize on execution device to avoid performing quantized ops on cpu
    device = get_execution_device(module)

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
            num_groups = math.ceil(weight_shape[1] / quantization_args.group_size)
            expected_shape = (weight_shape[0], max(num_groups, 1))

    scale_dtype = scale_dtype if scale_dtype is not None else module.weight.dtype
    # TODO: consider erroring out in the future as if the dtype if not one fo these,
    # there is likely bug

    if is_fp4(quantization_args=quantization_args) and base_name == "weight":
        scale_dtype = FP8_E4M3_DATA.dtype
        # When applying weight-only FP4 quantization, generate a global_scale
        # This scale is applied during runtime to ensure that the generated
        # local scale falls properly within the FP8 range (i.e max value is FP8_max)
        # which is the expected dtype of NVFP4A16 scales
        value = generate_global_scale(input_tensor=module.weight)
        value = value.to(device)
        init_global_scale = Parameter(value, requires_grad=False)
        register_offload_parameter(
            module, f"{base_name}_global_scale", init_global_scale
        )

    if scale_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ] and not is_fp4(quantization_args=quantization_args):
        scale_dtype = torch.float16

    # initializes empty scale, zero point, and g_idx parameters for the module
    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    register_offload_parameter(module, f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        if is_fp4(quantization_args=quantization_args):
            zp_dtype = FP8_E4M3_DATA.dtype
        else:
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
    register_offload_parameter(module, KVCacheScaleType.KEY.value, init_scale)

    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    register_offload_parameter(module, KVCacheScaleType.VALUE.value, init_scale)


# TODO: Potentially introduce an argument to turn this off
# Only relevant for NVFP4A16 currently
def update_fused_layer_weight_global_scales(model: torch.nn.Module):
    """
    When running NVFP4A16 quantization, update the global scale
    such that q,k,v layers are treated as one tensor with the same
    global_scale and gate_proj/up_proj layers are treated as one tensor
    with the same global scale. This is requirement currently being set
    by vLLM and may be removed in the future OR potentially make it
    an optional step.

    :param model: model to quantize
    """

    def _is_attention_module(module: Module):
        return "attention" in module.__class__.__name__.lower() and (
            hasattr(module, "k_proj")
            or hasattr(module, "v_proj")
            or hasattr(module, "qkv_proj")
        )

    def _is_mlp_module(module: Module):
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") or hasattr(module, "up_proj")
        )

    def _valid_fp4_quant(layer_list: List[torch.nn.Linear]):
        """
        Return True if all the linear layers in the layer_list are
        NVFP4A16 quantized.
        """
        for layer in layer_list:
            scheme = getattr(layer, "quantization_scheme", None)
            if scheme is None:
                return False

            weight_quant_args = scheme.weights

            if weight_quant_args is None:
                return False

            if not is_fp4(quantization_args=weight_quant_args):
                return False
        return True

    for name, submodule in iter_named_quantizable_modules(
        model,
        include_attn=True,
        include_mlp=True,
    ):

        if _is_attention_module(submodule):
            # already fused/treated as one layer
            if hasattr(submodule, "qkv_proj"):
                continue

            if not _valid_fp4_quant(
                [submodule.q_proj, submodule.v_proj, submodule.k_proj]
            ):
                continue

            q_weight = submodule.q_proj.weight.data
            v_weight = submodule.v_proj.weight.data
            k_weight = submodule.k_proj.weight.data

            value = generate_global_scale(
                input_tensor=torch.cat((q_weight, v_weight, k_weight), dim=0)
            )

            update_parameter_data(submodule.q_proj, value, "weight_global_scale")
            update_parameter_data(submodule.k_proj, value, "weight_global_scale")
            update_parameter_data(submodule.v_proj, value, "weight_global_scale")

        if _is_mlp_module(submodule):
            if not _valid_fp4_quant([submodule.gate_proj, submodule.up_proj]):
                continue

            gate_data = submodule.gate_proj.weight.data
            up_data = submodule.up_proj.weight.data

            value = generate_global_scale(
                input_tensor=torch.cat((gate_data, up_data), dim=0)
            )

            update_parameter_data(submodule.gate_proj, value, "weight_global_scale")
            update_parameter_data(submodule.up_proj, value, "weight_global_scale")
