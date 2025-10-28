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

import torch
from compressed_tensors.quantization.quant_args import BFLOAT16_DATA, FP4_E2M1_DATA


__all__ = ["convert_mxfp4_exp_scale", "generate_mxfp4_scales", "round_to_power_2"]

# Reference: https://github.com/vllm-project/vllm/blob/main/tests/quantization/reference_mxfp4.py # noqa: E501


def convert_mxfp4_exp_scale(
    scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Converts mxfp4 scales. Scales are powers of 2, with the
    exponents stored in uint8. Converts to dense dtype so that
    they can be applied to the weights and activations during QDQ

    :param scale: uint8 exponent scale
    :param dtype: dense dtype
    """
    assert scale.dtype == torch.uint8
    scale_exp = scale.to(torch.int32) - 127
    scale = 2.00 ** (scale_exp.to(torch.float))
    return scale.to(dtype)


def round_to_power_2(x: torch.Tensor) -> torch.Tensor:
    """
    Round values to the closest power of 2.
    This is done by masking the values with BFLOAT16_SIGN_EXPONENT_MASK
    which essentially removes the mantissa and keeps the exponent.
    i.e the closest power of 2 for the input_value.

    E.g:
        0.0825 = 1.32 (mantissa) x 2**-4 (exponent)
        0.0825 ==> -4 (exponent) + 127 = 123 = 01111011 (8 bits for bfloat16)
        0.0825 ==> 0.32 (mantissa) = 0101001 (7 bits for bfloat16)
        0.0825 == 0b01111011_0101001 (bfloat16)
        0b01111011_0101001 & 111111111_0000000 == 0b01111011_0000000
        Keep the exponent + sign bit to give you the closest power of 2, 0.0625

    :param x: tensor to round to closest power of 2
    """
    assert x.dtype == torch.bfloat16
    x = x.view(torch.uint16).to(torch.int32)

    # Find closest power of 2
    BFLOAT16_VAL_TO_ADD = 1 << (BFLOAT16_DATA.mantissa - FP4_E2M1_DATA.mantissa - 1)
    # Add value to push the value to the next exponent
    BFLOAT16_SIGN_EXPONENT_MASK = (
        (1 << (BFLOAT16_DATA.exponent + 1)) - 1
    ) << BFLOAT16_DATA.mantissa
    # mask to only keep exponent - we conservatively round down
    # to better represent smaller numbers / prevent overflow
    block_max_uint = torch.bitwise_and(
        x + BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK
    )
    return block_max_uint.to(torch.uint16).view(torch.bfloat16)


def generate_mxfp4_scales(x: torch.Tensor) -> torch.Tensor:
    """
    Generate mxfp4 scales. The scales require the following steps
    1. Round to the closest power of 2
    2. Convert to exponent
    3. Store in uint8

    Called when calculating qparams using observers.

    :param x: tensor to round to closest power of 2
    :returns uint8 scales as exponents
    """
    # Round to closest power of 2
    scale_power_2 = round_to_power_2(x)
    # Convert to exponent
    scale_exp = 127 + torch.floor(torch.log2(scale_power_2)).to(torch.int32) - 2
    # Clamp and store in uint8, as expected by mxfp4
    scale_exp = torch.clamp(
        scale_exp,
        max=torch.iinfo(torch.uint8).max,
        min=torch.iinfo(torch.uint8).min,
    )
    return scale_exp.to(torch.uint8)
