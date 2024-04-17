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

from typing import Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["calculate_qparams"]


def calculate_qparams(
    min_vals: Tensor, max_vals: Tensor, quantization_args: QuantizationArgs
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to caluclate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to caluclate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :return: tuple of the calculated scale(s) and zero point(s)
    """
    bit_range = 2**quantization_args.num_bits - 1
    if quantization_args.symmetric:
        symmetric_range = 2 * max(min_vals.abs(), max_vals.abs())
        scales = symmetric_range / bit_range
        zero_points = torch.tensor(0).to(torch.int8)
    else:
        # non-symmetric
        observed_range = max_vals - min_vals
        scales = observed_range / bit_range

        # scales from a 0 range should be set to 1
        scales[observed_range == 0] = 1

        zero_points = ((0 - min_vals) / scales).to(torch.int8)

    return scales, zero_points
