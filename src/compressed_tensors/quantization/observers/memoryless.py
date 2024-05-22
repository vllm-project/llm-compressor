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
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["MemorylessObserver"]


@Observer.register("memoryless", alias=["dynamic"])
class MemorylessObserver(Observer):
    """
    Implements a quantization observer that sets the scale and
    zero point based on the latest observed value without tracking state
    """

    def calculate_qparams(
        self,
        observed: Tensor,
        tensor_id: Optional[Any] = None,
        reduce_dims: Optional[Tuple[int]] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Returns the min and max values of observed tensor

        :param observed: observed tensor to calculate quantization parameters for
        :param tensor_id: optional id for tensor; not used for memoryless
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :return: tuple of scale and zero point derived from the observed tensor
        """

        if not reduce_dims:
            min_val, max_val = torch.aminmax(observed)
        else:
            min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        return calculate_qparams(min_val, max_val, self.quantization_args)
