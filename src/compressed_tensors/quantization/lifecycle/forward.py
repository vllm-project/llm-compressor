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
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Module


__all__ = ["wrap_module_forward_quantized", "maybe_calibrate_or_quantize"]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    bit_range = 2**args.num_bits
    q_max = torch.tensor(bit_range / 2 - 1, device=x.device)
    q_min = torch.tensor(-bit_range / 2, device=x.device)

    quantized_value = torch.clamp(
        torch.round(x / scale + zero_point),
        q_min,
        q_max,
    )

    if dtype is not None:
        quantized_value = quantized_value.to(dtype)

    return quantized_value


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (x_q - zero_point) * scale


@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    """
    Fake quantize the input tensor x depending on the group_size.
    if group_size is greater than 0, then q/dq by groups. The groups
    must be divisible by the column size
    if group_size is -1, then channel wise q/dq. THe input scale and
    zero_points are reshaped to support vectorization (Assumes 1 is
    the channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args that contain group_size info
    :return: fake quantized tensor

    """
    group_size = args.group_size

    # group
    if args.strategy == QuantizationStrategy.GROUP:

        DQ = torch.zeros_like(x)

        # TODO: vectorize the for loop
        # TODO: fix genetric assumption about the tensor size for computing group

        # TODO: make validation step for inputs

        while scale.ndim < 2:
            # pad scale and zero point dims for slicing
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1)

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
            zp = zero_point[:, i].view(-1, 1)

            idx = i * group_size
            Q = quantize(x[:, idx : (idx + group_size)], sc, zp, args)
            DQ[:, idx : (idx + group_size)] = dequantize(Q, sc, zp)

    # channel-wise
    elif args.strategy == QuantizationStrategy.CHANNEL:  # group_size == -1
        # before: scale shape = [channel_size]
        # after: scale shape = [1, channel_size]
        scale = scale.unsqueeze(0)
        zero_point = zero_point.unsqueeze(0)

        Q = quantize(x, scale, zero_point, args)
        DQ = dequantize(Q, scale, zero_point)

    # per-token
    elif args.strategy == QuantizationStrategy.TOKEN:
        # before: scale shape = [num_tokens]
        # after: scale shape = [num_tokens, 1]
        # x.shape = 1, num_tokens, 1]
        # scale gets broadcasted as expected withput having [1, num_tokens, 1] shape

        scale = scale.unsqueeze(1)
        zero_point = zero_point.unsqueeze(1)

        Q = quantize(x, scale, zero_point, args)
        DQ = dequantize(Q, scale, zero_point)

    else:
        Q = quantize(x, scale, zero_point, args)
        DQ = dequantize(Q, scale, zero_point)

    return DQ


def wrap_module_forward_quantized(module: Module, scheme: QuantizationScheme):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    forward_func_orig = module.forward.__func__

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
