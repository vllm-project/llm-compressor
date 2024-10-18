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

from functools import wraps
from math import ceil
from typing import Callable, Optional

import torch
from compressed_tensors.quantization.cache import QuantizedKVParameterCache
from compressed_tensors.quantization.observers.helpers import (
    calculate_range,
    compute_dynamic_scales_and_zp,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    round_to_quantized_type,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.utils import safe_permute, update_parameter_data
from torch.nn import Module


__all__ = [
    "quantize",
    "dequantize",
    "fake_quantize",
    "wrap_module_forward_quantized",
    "forward_quantize",
    "calibrate_activations",
]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Quantize the input tensor x using the QuantizationStrategy specified in args.
    Quantization can be done per tensor, channel, token or group. For group
    quantization, the group_size must be divisible by the column size. The input scale
    and zero_points are reshaped to support vectorization (Assumes 1 is the
    channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param dtype: optional dtype to cast the quantized output to
    :param g_idx: optional mapping from column index to group index
    :return: fake quantized tensor
    """

    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False,
        g_idx=g_idx,
    )


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    args: QuantizationArgs = None,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize a quantized input tensor x_q based on the strategy specified in args. If
    args is not provided, the strategy will be inferred.

    :param x: quantized input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args used to quantize x_q
    :param dtype: optional dtype to cast the dequantized output to
    :param g_idx: optional mapping from column index to group index
    :return: dequantized float tensor
    """
    if args is None:
        if scale.ndim == 0 or scale.ndim == 1:
            args = QuantizationArgs(strategy=QuantizationStrategy.TENSOR)
        elif scale.ndim == 2:
            if scale.shape[1] == 1:
                args = QuantizationArgs(strategy=QuantizationStrategy.CHANNEL)
            else:
                group_size = int(x_q.shape[1] / scale.shape[1])
                args = QuantizationArgs(
                    strategy=QuantizationStrategy.GROUP, group_size=group_size
                )
        else:
            raise ValueError(
                f"Could not infer a quantization strategy from scale with {scale.ndim} "
                "dimmensions. Expected 0 or 2 dimmensions."
            )

    if dtype is None:
        dtype = scale.dtype

    return _process_quantization(
        x=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=False,
        do_dequantize=True,
        dtype=dtype,
        g_idx=g_idx,
    )


@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fake quantize the input tensor x by quantizing then dequantizing with
    the QuantizationStrategy specified in args. Quantization can be done per tensor,
    channel, token or group. For group quantization, the group_size must be divisible
    by the column size. The input scale  and zero_points are reshaped to support
    vectorization (Assumes 1 is the channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param g_idx: optional mapping from column index to group index
    :return: fake quantized tensor
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        g_idx=g_idx,
    )


@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    do_quantize: bool = True,
    do_dequantize: bool = True,
) -> torch.Tensor:
    q_min, q_max = calculate_range(args, x.device)
    group_size = args.group_size

    if args.strategy == QuantizationStrategy.GROUP:
        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)
        columns = output.shape[1]

        # TODO: make validation step for inputs

        while scale.ndim < 2:
            # pad scale and zero point dims for slicing
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(
                    "tensor column shape must be divisble "
                    f"by the given group_size {group_size}"
                )

        # support column-order (default) quantization as well as other orderings
        # such as activation ordering. Below checks if g_idx has been initialized
        is_column_order = g_idx is None or -1 in g_idx
        if is_column_order:
            num_groups = int(ceil(columns / group_size))
            group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)

        else:
            group_indices, group_sizes = torch.unique(g_idx, return_counts=True)
            group_sizes = group_sizes[torch.argsort(group_indices)]

            perm = torch.argsort(g_idx)
            x = safe_permute(x, perm, dim=1)

        # TODO: experiment with vectorizing for loop for performance
        end = 0
        for index, group_count in enumerate(group_sizes):
            sc = scale[:, index].view(-1, 1)
            zp = zero_point[:, index].view(-1, 1) if zero_point is not None else None

            start = end
            end = start + group_count
            if do_quantize:
                output[:, start:end] = _quantize(
                    x[:, start:end],
                    sc,
                    zp,
                    q_min,
                    q_max,
                    args,
                    dtype=dtype,
                )

            if do_dequantize:
                input = output[:, start:end] if do_quantize else x[:, start:end]
                output[:, start:end] = _dequantize(input, sc, zp)

        if not is_column_order:
            output = safe_permute(output, torch.argsort(perm), dim=1)

    else:  # covers channel, token and tensor strategies
        if do_quantize:
            output = _quantize(
                x,
                scale,
                zero_point,
                q_min,
                q_max,
                args,
                dtype=dtype,
            )
        if do_dequantize:
            output = _dequantize(output if do_quantize else x, scale, zero_point)

    return output


def wrap_module_forward_quantized(module: Module, scheme: QuantizationScheme):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    if hasattr(module.forward, "__func__"):
        forward_func_orig = module.forward.__func__
    else:
        forward_func_orig = module.forward.func

    @wraps(forward_func_orig)  # ensures docstring, names, etc are propagated
    def wrapped_forward(self, *args, **kwargs):
        if not getattr(module, "quantization_enabled", True):
            # quantization is disabled on forward passes, return baseline
            # forward call
            return forward_func_orig.__get__(module, module.__class__)(*args, **kwargs)

        input_ = args[0]

        compressed = module.quantization_status == QuantizationStatus.COMPRESSED

        if scheme.input_activations is not None:
            # calibrate and (fake) quantize input activations when applicable
            # NOTE: will be moved out of compressed-tensors
            if (
                module.quantization_status == QuantizationStatus.CALIBRATION
                and not scheme.input_activations.dynamic
            ):
                calibrate_activations(
                    module=module,
                    value=input_,
                    base_name="input",
                    quantization_args=scheme.input_activations,
                )

            input_ = forward_quantize(module, input_, "input", scheme.input_activations)

        if scheme.weights is not None and not compressed:
            # calibrate and (fake) quantize weights when applicable
            unquantized_weight = self.weight.data.clone()
            self.weight.data = forward_quantize(
                module, self.weight, "weight", scheme.weights
            )

        # perform wrapped forward call
        output = forward_func_orig.__get__(module, module.__class__)(
            input_, *args[1:], **kwargs
        )
        if scheme.output_activations is not None:

            # calibrate and (fake) quantize output activations when applicable
            # kv_cache scales updated on model self_attn forward call in
            # wrap_module_forward_quantized_attn

            if (
                module.quantization_status == QuantizationStatus.CALIBRATION
                and not scheme.output_activations.dynamic
            ):
                calibrate_activations(
                    module=module,
                    value=output,
                    base_name="output",
                    quantization_args=scheme.ouput_activations,
                )

            output = forward_quantize(
                module, output, "output", scheme.output_activations
            )

        # restore back to unquantized_value
        if scheme.weights is not None and not compressed:
            self.weight.data = unquantized_weight

        return output

    # bind wrapped forward to module class so reference to `self` is correct
    bound_wrapped_forward = wrapped_forward.__get__(module, module.__class__)
    # set forward to wrapped forward
    setattr(module, "forward", bound_wrapped_forward)


def wrap_module_forward_quantized_attn(module: Module, scheme: QuantizationScheme):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    if hasattr(module.forward, "__func__"):
        forward_func_orig = module.forward.__func__
    else:
        forward_func_orig = module.forward.func

    @wraps(forward_func_orig)  # ensures docstring, names, etc are propagated
    def wrapped_forward(self, *args, **kwargs):

        # kv cache stored under weights
        if module.quantization_status == QuantizationStatus.CALIBRATION:
            quantization_args: QuantizationArgs = scheme.output_activations
            past_key_value: QuantizedKVParameterCache = quantization_args.get_kv_cache()
            kwargs["past_key_value"] = past_key_value

            # QuantizedKVParameterCache used for obtaining k_scale, v_scale only,
            # does not store quantized_key_states and quantized_value_state
            kwargs["use_cache"] = False

            attn_forward: Callable = forward_func_orig.__get__(module, module.__class__)

            past_key_value.reset_states()

            rtn = attn_forward(*args, **kwargs)

            update_parameter_data(
                module, past_key_value.k_scales[module.layer_idx], "k_scale"
            )
            update_parameter_data(
                module, past_key_value.v_scales[module.layer_idx], "v_scale"
            )

            return rtn

        return forward_func_orig.__get__(module, module.__class__)(*args, **kwargs)

    # bind wrapped forward to module class so reference to `self` is correct
    bound_wrapped_forward = wrapped_forward.__get__(module, module.__class__)
    # set forward to wrapped forward
    setattr(module, "forward", bound_wrapped_forward)


def calibrate_activations(
    module: Module,
    value: torch.Tensor,
    base_name: str,
    quantization_args: QuantizationArgs,
):
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return
    # calibration mode - get new quant params from observer
    if not hasattr(module, f"{base_name}_observer"):
        from compressed_tensors.quantization.lifecycle import initialize_observers

        initialize_observers(
            module=module, base_name=base_name, quantization_args=quantization_args
        )

    observer = getattr(module, f"{base_name}_observer")

    updated_scale, updated_zero_point = observer(value)

    # update scale and zero point
    update_parameter_data(module, updated_scale, f"{base_name}_scale")
    update_parameter_data(module, updated_zero_point, f"{base_name}_zero_point")


def forward_quantize(
    module: Module, value: torch.Tensor, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:

    # in compressed mode, the weight is already compressed and quantized so we don't
    # need to run fake quantization
    if (
        module.quantization_status == QuantizationStatus.COMPRESSED
        and base_name == "weight"
    ):
        return value

    if value.numel() == 0:
        # if the tensor is empty,
        # skip quantization
        return value

    g_idx = getattr(module, "weight_g_idx", None)

    if args.dynamic:
        # dynamic quantization - no need to invoke observer
        scale, zero_point = compute_dynamic_scales_and_zp(value=value, args=args)
    else:
        # static quantization - get previous scale and zero point from layer
        scale = getattr(module, f"{base_name}_scale")
        zero_point = getattr(module, f"{base_name}_zero_point", None)

    return fake_quantize(
        x=value,
        scale=scale,
        zero_point=zero_point,
        args=args,
        g_idx=g_idx,
    )


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:

    scaled = x / scale
    if zero_point is not None:
        scaled += zero_point.to(x.dtype)
    # clamp first because cast isn't guaranteed to be saturated (ie for fp8)
    clamped_value = torch.clamp(
        scaled,
        q_min,
        q_max,
    )
    quantized_value = round_to_quantized_type(clamped_value, args)
    if dtype is not None:
        quantized_value = quantized_value.to(dtype)

    return quantized_value


@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dequant_value = x_q.to(scale.dtype)

    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)
    dequant_value = dequant_value * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
