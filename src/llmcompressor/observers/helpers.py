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

from collections import Counter
from typing import Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils.helpers import calculate_range
from torch import FloatTensor, IntTensor, Tensor

__all__ = ["calculate_qparams", "get_observer_token_count"]


def get_observer_token_count(module: torch.nn.Module) -> Counter:
    """
    Parse the module and return the number of tokens observed by
    each module's observer.

    :param module: module to parse
    :return: counter with the number of tokens observed by each observer
    """
    token_counts = Counter()
    for name, module in module.named_modules():
        if name.endswith(".input_observer"):
            token_counts[name.replace(".input_observer", "")] = (
                module._num_observed_tokens
            )
    return token_counts


def calculate_qparams(
    min_vals: Tensor, max_vals: Tensor, quantization_args: QuantizationArgs
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to calculate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to calculate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :return: tuple of the calculated scale(s) and zero point(s)
    """
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    device = min_vals.device

    bit_min, bit_max = calculate_range(quantization_args, device)
    bit_range = bit_max - bit_min
    zp_dtype = quantization_args.pytorch_dtype()

    if quantization_args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        scales = max_val_pos / (float(bit_range) / 2)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        scales = (max_vals - min_vals) / float(bit_range)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = bit_min - (min_vals / scales)
        zero_points = torch.clamp(zero_points, bit_min, bit_max)

    # match zero-points to quantized type
    zero_points = zero_points.to(zp_dtype)

    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points
