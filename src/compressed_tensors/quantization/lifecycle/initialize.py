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
from typing import Optional, Tuple

import torch
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    ActivationOrdering,
    DynamicType,
    KVCacheScaleType,
    QuantizationArgs,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.utils import (
    is_fp4,
    is_kv_cache_quant_scheme,
    strategy_cdiv,
)
from compressed_tensors.utils import (
    disable_hf_hook,
    get_execution_device,
    register_offload_parameter,
)
from torch.nn import Module, Parameter


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
    "initialize_qparams",
]


_LOGGER = logging.getLogger(__name__)


def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
    force_zero_point: bool = True,
):
    """
    Attaches appropriate scales, zero points, and observers to a layer
    given its target quantization scheme.

    Previously initialized scales and zero points will be removed from
    module if they no longer apply to the scheme

    :param module: module to set for calibration
    :param scheme: scheme to use for quantization. if None is provided,
        will attempt to use scheme stored in the module under `quantization_scheme`,
        if not provided, the layer will be skipped
    :param force_zero_point: whether to force initialization of a zero point for
        symmetric quantization
    """
    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        return

    QuantizationMetadata.clear_all_qparams(module)

    if is_attention_module(module):
        # quantized actions based on calltime status
        _initialize_attn_scales(module)

    else:
        if not isinstance(module, torch.nn.Linear):
            _LOGGER.warning(f"Attempting to quantize module of type {type(module)}")

        # use weight to determine observed shapes and dtype
        if hasattr(module, "weight"):
            weight = module.weight
            assert isinstance(weight, torch.Tensor)
        else:
            # Note that a weight is required for both weight and activation
            # quantization in order to know the dtype of activation scales
            _LOGGER.warning(
                f"module type {type(module)} targeted for quantization but "
                f"has no attribute weight, skipping quantization for {type(module)}"
            )
            return

        if scheme.input_activations is not None:
            initialize_qparams(
                module,
                "input",
                scheme.input_activations,
                observed_shape=weight.shape[-1:],
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
            )

        if scheme.weights is not None:
            initialize_qparams(
                module,
                "weight",
                scheme.weights,
                observed_shape=weight.shape,
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
            )

        output_is_kv_cache = is_kv_cache_quant_scheme(scheme)
        if scheme.output_activations is not None and not output_is_kv_cache:
            initialize_qparams(
                module,
                "output",
                scheme.output_activations,
                observed_shape=weight.shape[:-1],
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
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


def initialize_qparams(
    module: Module,
    base_name: str,
    quantization_args: QuantizationArgs,
    observed_shape: Tuple[int],
    observed_dtype: torch.dtype,
    force_zero_point: bool = True,
):
    """
    Initialize quantization parameters for a given basename according to the passed
    quantization args. The shape and dtype of the observed weight/activation must also
    be provided.

    Scales will always be initialized. Global scales are initialized depending on args.
    Zero points will be initialized if not symmetric or if `force_zero_point` is True.

    :param module: module to register qparams to
    :param base_name: base name of qparams, for example "input", "weight", "k", "v"
    :param quantization_args: arguments for quantization
    :param observed_shape: last (right-most) known dimensions of the observed weight/act
    :param observed_dtype: dtype of the observed weight/actt
    :param force_zero_point: force the zero_point parameter to be initialized
    """
    strategy = quantization_args.strategy
    dynamic = quantization_args.dynamic
    actorder = quantization_args.actorder
    device = get_execution_device(module)  # avoid performing intialization ops on cpu

    # Skip all intialization for fully dynamic quantization
    if dynamic is True:
        return

    # 0. Create global scale for tensor-group quantization
    if strategy == QuantizationStrategy.TENSOR_GROUP:
        init_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        register_offload_parameter(
            module, f"{base_name}_global_scale", init_global_scale
        )

    # Skip scale/zp initialization for locally dynamic quantization
    if dynamic == DynamicType.LOCAL:
        return

    # 1. Infer expected scale/zp shape
    if strategy == QuantizationStrategy.TENSOR:
        expected_shape = (1,)

    elif strategy == QuantizationStrategy.TOKEN:
        expected_shape = (1, 1)

    elif strategy == QuantizationStrategy.CHANNEL:
        if len(observed_shape) < 2:
            raise ValueError("Channel quant requires at least 2 observed dimensions")

        expected_shape = (observed_shape[-2], 1)

    elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        assert quantization_args.group_size is not None
        if len(observed_shape) < 1:
            raise ValueError("Group quant requires at least 1 observed dimension")

        group_size = quantization_args.group_size
        num_groups = strategy_cdiv(observed_shape[-1], group_size, strategy)
        expected_shape = (*observed_shape[:-1], num_groups)

        # initialize activation ordering if applicable
        if actorder == ActivationOrdering.GROUP:
            init_g_idx = Parameter(
                torch.full((observed_shape[-1],), -1, device=device, dtype=torch.int),
                requires_grad=False,
            )
            register_offload_parameter(module, f"{base_name}_g_idx", init_g_idx)

    elif strategy == QuantizationStrategy.BLOCK:
        assert quantization_args.block_structure is not None
        if len(observed_shape) < 2:
            raise ValueError("Block quant requires at least 2 observed dimensions")

        block_structure = quantization_args.block_structure
        num_rows = strategy_cdiv(observed_shape[-2], block_structure[-2], strategy)
        num_cols = strategy_cdiv(observed_shape[-1], block_structure[-1], strategy)
        expected_shape = (num_rows, num_cols)

    else:
        assert False, f"Unknown strategy {strategy}"

    # 2. Identify quantization scale and zp dtype
    scale_dtype = observed_dtype

    if is_fp4(quantization_args=quantization_args):
        scale_dtype = zp_dtype = FP8_E4M3_DATA.dtype
    else:
        # TODO: consider erroring out in the future as if the dtype if not one of these,
        # there is likely bug
        if scale_dtype not in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]:
            scale_dtype = torch.bfloat16
        zp_dtype = quantization_args.pytorch_dtype()

    # 3. Initializes scale/zp for the module
    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    register_offload_parameter(module, f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        init_zero_point = Parameter(
            torch.zeros(expected_shape, device=device, dtype=zp_dtype),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_zero_point", init_zero_point)


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
