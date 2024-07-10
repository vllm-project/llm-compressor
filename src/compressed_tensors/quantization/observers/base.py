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

import logging
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.registry.registry import RegistryMixin
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module


_LOGGER = logging.getLogger(__name__)


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
        self._num_observed_tokens = None

    @torch.no_grad()
    def forward(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        self.record_observed_tokens(observed)
        return self.get_qparams(observed=observed)

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
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
                group_idxs = range(0, columns, self.quantization_args.group_size)
                for group_id, group_idx in enumerate(group_idxs):
                    scale, zero_point = self.get_qparams_along_dim(
                        observed[:, group_idx : (group_idx + group_size)],
                        0,
                        tensor_id=group_id,
                    )
                    scales.append(scale)
                    zero_points.append(zero_point)

                self._scale = torch.cat(scales, dim=1, out=self._scale)
                self._zero_point = torch.cat(zero_points, dim=1, out=self._zero_point)

            elif self.quantization_args.strategy == QuantizationStrategy.CHANNEL:
                # assume observed is transposed, because its the output, hence use dim 0
                self._scale, self._zero_point = self.get_qparams_along_dim(observed, 0)

            elif self.quantization_args.strategy == QuantizationStrategy.TOKEN:
                # use dim 1, assume the obsersed.shape = [batch, token, hidden]
                # should be batch, token
                self._scale, self._zero_point = self.get_qparams_along_dim(
                    observed,
                    dim={0, 1},
                )

        return self._scale, self._zero_point

    def get_qparams_along_dim(
        self,
        observed,
        dim: Union[int, Iterable[int]],
        tensor_id: Optional[Any] = None,
    ):
        dim = set(dim)

        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx not in dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims, tensor_id=tensor_id
        )

    def record_observed_tokens(self, batch_tensor: Tensor):
        """
        Counts the number of tokens observed during the
        forward passes. The count is aggregated in the
        _num_observed_tokens attribute of the class.

        Note: The batch_tensor is expected to have two dimensions
            (batch_size * sequence_length, num_features). This is the
            general shape expected by the forward pass of the expert
            layers in a MOE model. If the input tensor does not have
            two dimensions, the _num_observed_tokens attribute will be set
            to None.
        """
        if not isinstance(batch_tensor, Tensor):
            raise ValueError(f"Expected value to be a tensor, got {type(batch_tensor)}")

        if batch_tensor.ndim != 2:
            _LOGGER.debug(
                "The input tensor is expected to have two dimensions "
                "(batch_size * sequence_length, num_features). "
                f"The input tensor has {batch_tensor.ndim} dimensions."
            )
            return

        if self._num_observed_tokens is None:
            # initialize the count
            self._num_observed_tokens = 0

        # batch_tensor (batch_size * sequence_length, num_features)
        # observed_tokens (batch_size * sequence_length)
        observed_tokens, _ = batch_tensor.shape
        self._num_observed_tokens += observed_tokens
