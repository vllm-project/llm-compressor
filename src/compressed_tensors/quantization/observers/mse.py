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
from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.observers.helpers import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["MovingAverageMSEObserver"]


@Observer.register("mse")
class MovingAverageMSEObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on a moving average of the mse-clipped min and max observed values
    """

    def __init__(
        self,
        quantization_args: QuantizationArgs,
        averaging_constant: float = 0.01,
        grid: float = 100.0,
        maxshrink: float = 0.80,
        norm: float = 2.4,
    ):
        super().__init__(quantization_args=quantization_args)

        self.min_val = {}
        self.max_val = {}
        self.averaging_constant = averaging_constant
        self.grid = grid
        self.maxshrink = maxshrink
        self.norm = norm

    def calculate_mse_min_max(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ):
        """
        Computes the mse-clipped min and max values of the observed tensor by
        optimizing for quantization error

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned values will be shaped (1,) along the reduced dimensions
        :return: tuple of min and max values derived from the observed tensor
        """
        from compressed_tensors.quantization.lifecycle import fake_quantize

        if not reduce_dims:
            absolute_min_val, absolute_max_val = torch.aminmax(observed)
        else:
            absolute_min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            absolute_max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        best = torch.full(absolute_min_val.shape, float("inf"))
        min_val = torch.ones(absolute_min_val.shape)
        max_val = torch.zeros(absolute_max_val.shape)
        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            shrinked_min_val = p * absolute_min_val
            shrinked_max_val = p * absolute_max_val

            candidate_scales, candidate_zero_points = calculate_qparams(
                shrinked_min_val, shrinked_max_val, self.quantization_args
            )
            q = fake_quantize(
                observed,
                candidate_scales,
                candidate_zero_points,
                self.quantization_args,
            )

            q -= observed
            q.abs_()
            q.pow_(self.norm)
            if not reduce_dims:
                err = torch.sum(q)
            else:
                err = torch.sum(q, reduce_dims, keepdims=True)

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                min_val[tmp] = shrinked_min_val[tmp]
                max_val[tmp] = shrinked_max_val[tmp]
        return min_val, max_val

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Updates the mse-clipped min and max values of the observed tensor using
        a moving average smoothed by the averaging_constant

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: tuple of scale and zero point derived from the observed tensor
        """
        min_val, max_val = self.calculate_mse_min_max(observed, reduce_dims)

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

        tensor_id = tensor_id or "default"
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
