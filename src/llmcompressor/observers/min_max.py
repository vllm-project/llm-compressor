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

from typing import Any, Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from torch import FloatTensor, IntTensor, Tensor

from llmcompressor.observers.base import Observer

__all__ = ["MovingAverageMinMaxObserver"]


@Observer.register("minmax")
class MovingAverageMinMaxObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on a moving average of the overall min and max observed values
    """

    def __init__(
        self, quantization_args: QuantizationArgs, averaging_constant: float = 0.01
    ):
        super().__init__(quantization_args=quantization_args)

        self.min_val = {}
        self.max_val = {}
        self.averaging_constant = averaging_constant

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Updates the observed min and max using a moving average smoothed by the
        averaging_constant

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: tuple of scale and zero point derived from the observed tensor
        """
        tensor_id = tensor_id or "default"

        if not reduce_dims:
            min_val, max_val = torch.aminmax(observed)
        else:
            min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        running_min_val = self.min_val.get(tensor_id, None)
        running_max_val = self.max_val.get(tensor_id, None)

        if running_min_val is None or running_max_val is None:
            updated_min_val = min_val
            updated_max_val = max_val
        else:
            updated_min_val = running_min_val + self.averaging_constant * (
                min_val - running_min_val
            )
            updated_max_val = running_max_val + self.averaging_constant * (
                max_val - running_max_val
            )

        self.min_val[tensor_id] = updated_min_val
        self.max_val[tensor_id] = updated_max_val

        return calculate_qparams(
            updated_min_val, updated_max_val, self.quantization_args
        )

    def get_qparams_along_dim(
        self, observed, dim: int, tensor_id: Optional[Any] = None
    ):
        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx != dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims, tensor_id=tensor_id
        )

    def reset(self):
        """
        Reset the state of the observer, including min and maximum values
        """
        super().reset()
        self.min_val = {}
        self.max_val = {}
