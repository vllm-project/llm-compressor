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
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from compressed_tensors.transform.utils.utils import (
    apply_transform_weight,
    get_matrix_size,
)
from compressed_tensors.utils import get_offloaded_device
from compressed_tensors.utils.helpers import ParameterizedDefaultDict
from torch import Tensor, device, dtype
from torch.nn import Linear, Module, Parameter


@TransformFactory.register("hadamard")
class HadamardFactory(TransformFactory):
    """
    Factory used to apply hadamard transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    def __init__(self, name: str, scheme: TransformScheme, seed: Optional[int] = None):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)

    def create_transform(self, module: Module, args: TransformArgs):
        """
        Create a HadamardTransform for applying to a module. Transforms with the same
        size, dtype, and device are cached

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        """
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args.location)
        dtype = module.weight.dtype
        device = get_offloaded_device(module)

        weight = self.weights[size, dtype, device]
        return HadamardTransform(weight, args)

    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        data = deterministic_hadamard_matrix(size, dtype, device)
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)


class HadamardTransform(TransformBase):
    def __init__(self, weight: Parameter, args: TransformArgs):
        super().__init__()
        self.weight = weight
        self.args = args

    def forward(self, value: Tensor) -> Tensor:
        if not self.args.inverse:
            weight = self.weight
        else:
            weight = self.weight.T

        return apply_transform_weight(weight, value, self.args.location)
