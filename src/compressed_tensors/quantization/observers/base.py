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

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.registry.registry import RegistryMixin
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

    @torch.no_grad()
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

    def post_calculate_qparams(self) -> None:
        """
        Run any logic specific to its observers after running calculate_qparams
        """
        ...

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
            group_size = self.quantization_args.group_size

            if self.quantization_args.strategy == QuantizationStrategy.TENSOR:

                # re-calculate scale and zero point, update the stored value
                self._scale, self._zero_point = self.calculate_qparams(observed)

            elif self.quantization_args.strategy == QuantizationStrategy.GROUP:
                columns = observed.shape[1]
                scales, zero_points = [], []
                for i in range(0, columns, self.quantization_args.group_size):
                    scale, zero_point = self.get_qparams_along_dim(
                        observed[:, i : (i + group_size)],
                        0,
                    )
                    scales.append(scale)
                    zero_points.append(zero_point)
                self._scale = torch.stack(scales, dim=1, out=self._scale)
                self._zero_point = torch.stack(zero_points, dim=1, out=self._zero_point)

            elif self.quantization_args.strategy == QuantizationStrategy.CHANNEL:
                # assume observed is transposed, because its the output, hence use dim 0
                self._scale, self._zero_point = self.get_qparams_along_dim(observed, 0)

            elif self.quantization_args.strategy == QuantizationStrategy.TOKEN:

                # use dim 1, assume the obsersed.shape = [batch, token, hidden]
                # should be batch, token

                self._scale, self._zero_point = self.get_qparams_along_dim(
                    observed, dim=1
                )

        return self._scale, self._zero_point

    def get_qparams_along_dim(self, observed, dim: int):
        # TODO: add documentation that specifies the shape must
        #   be padded with 1-dims so the scales are along the right channel
        # TODO: generalize the logic for reduce_dims
        scales, zero_points = [], []

        # TODO: make a more generic way to get the channel
        num_dims = observed.shape[dim]

        for dim_idx in range(num_dims):
            scale, zero_point = self.calculate_qparams(
                observed.select(dim=dim, index=dim_idx)
            )

            scales.append(scale)
            zero_points.append(zero_point)
        # breakpoint()
        return torch.stack(scales), torch.stack(zero_points)
