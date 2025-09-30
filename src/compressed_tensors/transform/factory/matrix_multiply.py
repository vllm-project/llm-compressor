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

from typing import Optional

import torch
from compressed_tensors.transform import TransformArgs, TransformScheme
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.transform.utils.matrix import (
    apply_transform_weight,
    get_transform_size,
)
from compressed_tensors.utils.helpers import ParameterizedDefaultDict
from compressed_tensors.utils.offload import get_offloaded_device
from torch import Tensor, device, dtype
from torch.nn import Module, Parameter


@TransformFactory.register("random-matrix")
class RandomMatrixFactory(TransformFactory):
    """
    Factory used to apply random matrix transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    def __init__(self, name: str, scheme: TransformScheme, seed: Optional[int] = None):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.inverses = ParameterizedDefaultDict(self._create_inverse)

    def create_transform(self, module: Module, args: TransformArgs):
        """
        Create a RandomMatrixTransform for applying to a module. Transforms with the
        same size, dtype, and device are cached

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        """
        assert hasattr(module, "weight")
        size = get_transform_size(module, args.location, self.scheme.head_dim)
        device = get_offloaded_device(module)
        precision = self.scheme.precision if args.is_online() else torch.float64

        factory_kwargs = {"device": device, "precision": precision}
        weight = self.weights.get(size, factory_kwargs=factory_kwargs)
        if args.inverse:
            weight = self.inverses[weight]

        return RandomMatrixTransform(weight, self.scheme, args, type(module))

    def _create_weight(self, size: int, device: device, precision: dtype) -> Parameter:
        # TODO: construct such that weight is invertible (has non-zero determinant)
        data = torch.rand(
            (size, size),
            generator=self.generator,
            dtype=precision,
            device=device,
        )
        return Parameter(data, requires_grad=self.scheme.requires_grad)

    def _create_inverse(self, weight: Parameter) -> Parameter:
        data = high_precision_invert(weight.data)
        data = data.contiguous()  # ensure proper serialization
        return Parameter(data, requires_grad=False)


class RandomMatrixTransform(TransformBase):
    def __init__(
        self,
        weight: Tensor,
        scheme: TransformScheme,
        args: TransformArgs,
        module_type: type[torch.nn.Module],
    ):
        super().__init__()
        self.weight = weight  # is an inverse if args.inverse
        self.scheme = scheme
        self.args = args
        self.module_type = module_type

    def forward(self, value: Tensor) -> Parameter:
        return apply_transform_weight(
            self.weight.to(device=value.device),
            value.to(dtype=self.weight.dtype),
            self.args.location,
            self.module_type,
        ).to(value.dtype)

    def right_inverse(self, value: Tensor) -> Tensor:
        inverse = high_precision_invert(self.weight)
        return apply_transform_weight(
            inverse.to(device=value.device),
            value.to(dtype=inverse.dtype),
            self.args.location,
            self.module_type,
        ).to(value.dtype)


def high_precision_invert(weight: Tensor) -> Tensor:
    return torch.linalg.inv(weight.to(torch.float64)).to(weight.dtype)
