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
from sparsetensors.quantization.observers.base import Observer
from sparsetensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["MinMaxObserver"]


@Observer.register("minmax")
class MinMaxObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on the latest observed value
    """

    def __init__(self, quantization_args: QuantizationArgs):
        super().__init__(quantization_args=quantization_args)

        self.min_val = float("inf")
        self.max_val = -float("inf")
        self.counter = 0

    def calculate_qparams(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :return: tuple of scale and zero point derived from the observed tensor
        """
        # TODO: Add support for full range of quantization Args, only supports 8bit
        #       per tensor
        bit_range = 255
        min_val = torch.tensor([observed.min()])
        max_val = torch.tensor([observed.max()])

        # update running average
        if self.counter > 0:
            # self.min_val = (self.min_val * self.counter + min_val) / (self.counter + 1)
            # self.max_val = (self.max_val * self.counter + max_val) / (self.counter + 1)
            self.min_val = torch.min(min_val, self.min_val)
            self.max_val = torch.max(max_val, self.max_val)
        else:
            self.min_val = min_val
            self.max_val = max_val

        # ensure that the zeros are in the range
        min_val = torch.min(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.max(self.max_val, torch.zeros_like(self.max_val))

        self.counter += 1

        if self.quantization_args.symmetric:
            symmetric_range = 2 * max(min_val.abs(), max_val.abs())
            scale = symmetric_range / bit_range
            zero_point = torch.tensor(0).to(torch.int8)
        else:
            # non-symmetric
            observed_range = max_val - min_val
            scale = observed_range / bit_range

            # scales from a 0 range should be set to 1
            scale[observed_range == 0] = 1

            zero_point = ((0 - min_val) / scale).to(torch.int8)

        return scale, zero_point
