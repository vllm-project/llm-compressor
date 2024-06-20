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
from typing import Optional

import torch
from compressed_tensors.quantization.observers.helpers import calculate_range
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    round_to_quantized_type,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Module


__all__ = [
    "quantize",
    "dequantize",
    "fake_quantize",
    "wrap_module_forward_quantized",
    "maybe_calibrate_or_quantize",
]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
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
    :return: fake quantized tensor
    """
    # ensure all tensors are on the same device
    # assumes that the target device is the input
    # tensor's device
    if x.device != scale.device:
        scale = scale.to(x.device)
    if x.device != zero_point.device:
        zero_point = zero_point.to(x.device)

    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False,
    )


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    args: QuantizationArgs = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize a quantized input tensor x_q based on the strategy specified in args. If
    args is not provided, the strategy will be inferred.

    :param x: quantized input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args used to quantize x_q
    :param dtype: optional dtype to cast the dequantized output to
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
    )


@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
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
    :return: fake quantized tensor
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=True,
        do_dequantize=True,
    )


@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
    do_quantize: bool = True,
    do_dequantize: bool = True,
) -> torch.Tensor:

    q_min, q_max = calculate_range(args, x.device)
    group_size = args.group_size

    if args.strategy == QuantizationStrategy.GROUP:
        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)

        # TODO: vectorize the for loop
        # TODO: fix genetric assumption about the tensor size for computing group

        # TODO: make validation step for inputs

        while scale.ndim < 2:
            # pad scale and zero point dims for slicing
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

        columns = x.shape[1]
        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(
                    "tesnor column shape must be divisble "
                    f"by the given group_size {group_size}"
                )
        for i in range(ceil(columns / group_size)):
            # scale.shape should be [nchan, ndim]
            # sc.shape should be [nchan, 1] after unsqueeze
            sc = scale[:, i].view(-1, 1)
            zp = zero_point[:, i].view(-1, 1) if zero_point is not None else None

            idx = i * group_size
            if do_quantize:
                output[:, idx : (idx + group_size)] = _quantize(
                    x[:, idx : (idx + group_size)],
                    sc,
                    zp,
                    q_min,
                    q_max,
                    args,
                    dtype=dtype,
                )
            if do_dequantize:
                input = (
                    output[:, idx : (idx + group_size)]
                    if do_quantize
                    else x[:, idx : (idx + group_size)]
                )
                output[:, idx : (idx + group_size)] = _dequantize(input, sc, zp)

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
        input_ = args[0]

        if scheme.input_activations is not None:
            # calibrate and (fake) quantize input activations when applicable
            input_ = maybe_calibrate_or_quantize(
                module, input_, "input", scheme.input_activations
            )

        if scheme.weights is not None:
            # calibrate and (fake) quantize weights when applicable
            unquantized_weight = self.weight.data.clone()
            self.weight.data = maybe_calibrate_or_quantize(
                module, self.weight, "weight", scheme.weights
            )

        # perform wrapped forward call
        output = forward_func_orig.__get__(module, module.__class__)(
            input_, *args[1:], **kwargs
        )

        if scheme.output_activations is not None:
            # calibrate and (fake) quantize output activations when applicable
            output = maybe_calibrate_or_quantize(
                module, output, "output", scheme.output_activations
            )

        # restore back to unquantized_value
        if scheme.weights is not None:
            self.weight.data = unquantized_weight

        return output

    # bind wrapped forward to module class so reference to `self` is correct
    bound_wrapped_forward = wrapped_forward.__get__(module, module.__class__)
    # set forward to wrapped forward
    setattr(module, "forward", bound_wrapped_forward)


def maybe_calibrate_or_quantize(
    module: Module, value: torch.Tensor, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:
    # only run quantized for the included stages
    if module.quantization_status not in {
        QuantizationStatus.CALIBRATION,
        QuantizationStatus.FROZEN,
    }:
        return value

    if args.dynamic:
        # dynamic quantization - get scale and zero point directly from observer
        observer = getattr(module, f"{base_name}_observer")
        scale, zero_point = observer(value)
    else:
        # static quantization - get previous scale and zero point from layer
        scale = getattr(module, f"{base_name}_scale")
        zero_point = getattr(module, f"{base_name}_zero_point")

        if module.quantization_status == QuantizationStatus.CALIBRATION:
            # calibration mode - get new quant params from observer
            observer = getattr(module, f"{base_name}_observer")

            updated_scale, updated_zero_point = observer(value)

            # update scale and zero point
            device = next(module.parameters()).device
            scale.data = updated_scale.to(device)
            zero_point.data = updated_zero_point.to(device)
    return fake_quantize(value, scale, zero_point, args)


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

    scaled = x / scale + zero_point.to(x.dtype)
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

    dequant_value = x_q
    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)
    dequant_value = dequant_value.to(scale.dtype) * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
