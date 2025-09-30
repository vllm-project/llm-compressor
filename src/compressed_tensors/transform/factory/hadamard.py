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

from typing import List, Optional

import torch
from compressed_tensors.transform import TransformArgs, TransformScheme
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from compressed_tensors.transform.utils.matrix import (
    apply_transform_weight,
    get_transform_size,
)
from compressed_tensors.utils import get_execution_device, get_offloaded_device
from compressed_tensors.utils.helpers import ParameterizedDefaultDict
from torch import Tensor, device, dtype
from torch.nn import Module, Parameter


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
        self.perms = ParameterizedDefaultDict(self._create_permutation)

    def create_transform(self, module: Module, args: TransformArgs):
        """
        Create a HadamardTransform for applying to a module. Transforms with the same
        size, dtype, and device are cached

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        """
        assert hasattr(module, "weight")
        size = get_transform_size(module, args.location, self.scheme.head_dim)
        exec_device = get_execution_device(module)
        device = get_offloaded_device(module)
        precision = self.scheme.precision if args.is_online() else torch.float64

        factory_kwargs = {
            "device": device,
            "construct_device": exec_device,
            "precision": precision,
        }
        weight = self.weights.get(size, factory_kwargs=factory_kwargs)
        # TODO: permutations should be keyed by fused modules, not weight
        perm = self.perms[weight] if self.scheme.randomize else None
        return HadamardTransform(weight, perm, self.scheme, args, type(module))

    def _create_weight(
        self,
        size: int,
        device: device,
        construct_device: device,
        precision: dtype,
    ) -> Parameter:
        data = deterministic_hadamard_matrix(size, precision, construct_device)
        data = data.to(device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)

    def _create_permutation(self, weight: Parameter) -> Parameter:
        data = torch.randperm(weight.size(0), generator=self.generator)
        return Parameter(data, requires_grad=False)


class HadamardTransform(TransformBase):
    _dynamic_tied_weights_keys: List[str] = ["weight", "perm"]

    def __init__(
        self,
        weight: Parameter,
        perm: Optional[Parameter],
        scheme: TransformScheme,
        args: TransformArgs,
        module_type: type[torch.nn.Module],
    ):
        super().__init__()
        self.weight = weight
        self.perm = perm
        self.scheme = scheme
        self.args = args
        self.module_type = module_type
        self._scale = torch.tensor(weight.size(0), dtype=torch.float64).sqrt()

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight

        if self.perm is not None:
            weight = weight[self.perm][:, self.perm]

        if self.args.inverse:
            weight = weight.T

        return (
            apply_transform_weight(
                weight.to(device=value.device),
                value.to(dtype=weight.dtype),
                self.args.location,
                self.module_type,
            )
            / self._scale
        ).to(value.dtype)
