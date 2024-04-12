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

from typing import Optional, Tuple

from sparsetensors.quantization.quant_args import QuantizationArgs
from sparsetensors.registry.registry import RegistryMixin
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module


__all__ = ["Observer"]


class Observer(Module, RegistryMixin):
    """
    Base Observer class to be subclassed for specific implementation.
    Subclasses should override `calculate_qparams` to return a scale, zero_point
    pair
    """

    def __init__(self, quantization_args: QuantizationArgs):
        self.quantization_args: QuantizationArgs = quantization_args
        super().__init__()
        self._scale = None
        self._zero_point = None

    def forward(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        return self.get_qparams(observed=observed)

    def calculate_qparams(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :return: tuple of scale and zero point derived from the observed tensor
        """
        raise NotImplementedError(f"{self.__class__} must implement calculate_qparams")

    def get_qparams(
        self, observed: Optional[Tensor] = None
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Convenience function to wrap overwritten calculate_qparams
        adds support to make observed tensor optional and support for tracking latest
        calculated scale and zero point
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        if observed is not None:
            # re-calcualte scale and zero point, update the stored value
            self._scale, self._zero_point = self.calculate_qparams(observed)
        return self._scale, self._zero_point
