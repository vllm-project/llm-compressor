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

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Module


__all__ = ["wrap_module_forward_quantized"]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
) -> torch.Tensor:
    return torch.clamp(
        torch.round(
            x / scale + zero_point,
        ),
        q_min,
        q_max,
    )


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
    bit_range = 2**args.num_bits
    max_q = torch.tensor(bit_range / 2 - 1, device=x.device)
    min_q = torch.tensor(-bit_range / 2, device=x.device)
    Q = torch.zeros_like(x)
    Q = quantize(x, scale, zero_point, min_q, max_q)
    return dequantize(Q, scale, zero_point)


def wrap_module_forward_quantized(module: Module, scheme: QuantizationScheme):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    forward_func_orig = module.forward.__func__

    @wraps(forward_func_orig)  # ensures docstring, names, etc are propagated
    def wrapped_forward(self, *args, **kwargs):
        input_ = args[0]

        if scheme.input_activations is not None:
            # calibrate and (fake) quantize input activations when applicable
            input_ = _maybe_calibrate_or_quantize(
                module, input_, "input", scheme.input_activations
            )

        if scheme.weights is not None:
            # calibrate and (fake) quantize weights when applicable
            unquantized_weight = self.weight.data.clone()
            self.weight.data = _maybe_calibrate_or_quantize(
                module, self.weight, "weight", scheme.weights
            )

        # perform wrapped forward call
        output = forward_func_orig.__get__(module, module.__class__)(
            input_, *args[1:], **kwargs
        )

        if scheme.output_activations is not None:
            # calibrate and (fake) quantize output activations when applicable
            output = _maybe_calibrate_or_quantize(
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


def _maybe_calibrate_or_quantize(
    module: Module, value: Module, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:
    # only run quantized for the included stages
    if module.quantization_status not in {
        QuantizationStatus.CALIBRATION,
        QuantizationStatus.FROZEN,
    }:
        return value

    device = next(module.parameters()).device
    scale = getattr(module, f"{base_name}_scale")
    zero_point = getattr(module, f"{base_name}_zero_point")

    if module.quantization_status == QuantizationStatus.CALIBRATION:
        # get observer and get new quant params from observation
        observer = getattr(module, f"{base_name}_observer")
        updated_scale, updated_zero_point = observer(value)

        # update scale and zero point
        scale.data = updated_scale.to(device)
        zero_point.data = updated_zero_point.to(device)

    return fake_quantize(value, scale, zero_point, args)
