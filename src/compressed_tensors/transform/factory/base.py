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

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.utils.parametrize as P
from compressed_tensors.quantization.lifecycle import is_target  # TODO: move to utils
from compressed_tensors.registry.registry import RegistryMixin, T
from compressed_tensors.transform import (
    TransformArgs,
    TransformLocation,
    TransformScheme,
)
from compressed_tensors.utils import (
    align_module_device,
    has_offloaded_params,
    patch_attr,
    register_offload_module,
    update_offload_parameter,
)
from torch import Tensor
from torch.nn import Module, Parameter


__all__ = ["TransformFactory", "TransformBase"]


class TransformFactory(RegistryMixin, ABC):
    """
    Abstract factory base used to create and apply transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    def __init__(self, name: str, scheme: TransformScheme, seed: Optional[int] = None):
        self.name = name
        self.scheme = scheme
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    @classmethod
    def from_scheme(cls: type[T], scheme: TransformScheme, **kwargs) -> T:
        """
        Create a transform factory from a scheme

        :param scheme: defines how transforms should be created
        :param kwargs: TransformFactory constructor arguments
        :return: subclass of `TransformFactory` corresponding to the scheme type
        """
        constructor = cls.get_value_from_registry(name=scheme.type)
        return constructor(scheme=scheme, **kwargs)

    @abstractmethod
    def create_transform(self, module: Module, args: TransformArgs) -> "TransformBase":
        """
        Abstract method which defines how a transform should be created. May utilize
        caching to maximize shared memory

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        :return: instance of TransformBase
        """
        raise NotImplementedError()

    def apply_to_model(self, model: Module):
        """
        Create transforms and apply them to the model

        :param model: module to apply transforms to
        """
        for arg in self.scheme.apply:
            for name, module in list(model.named_modules()):
                if is_target(name, module, arg.targets, arg.ignore):
                    self._apply_to_module(module, arg)

    def _apply_to_module(self, module: Module, args: TransformArgs):
        """
        Create transforms and apply them to the module

        :param module: target module to apply transforms to
        :param args: defines how the transform will be applied to the target module
        """
        # create transform as submodule
        transform_name = f"{self.name}_{args.location}"
        transform = self.create_transform(module, args)
        register_offload_module(module, transform_name, transform)  # (1)

        # register input transformation hook
        if args.location == TransformLocation.INPUT:

            def input_hook(_, args):
                input = args[0]
                return transform(input)

            module.register_forward_pre_hook(input_hook, prepend=True)

        # eagerly apply transformation to weight
        elif args.location in (
            TransformLocation.WEIGHT_INPUT,
            TransformLocation.WEIGHT_OUTPUT,
        ):
            assert isinstance(module, torch.nn.Linear)
            assert module.bias is None

            with torch.no_grad(), align_module_device(module):
                update_offload_parameter(module, "weight", transform(module.weight))

            if self.scheme.requires_grad:
                # for training, the weight changes with every forward pass
                # so we can leverage parametrization to propagate the gradient
                if has_offloaded_params(module):
                    raise ValueError("Offloaded training is not supported")
                P.register_parametrization(module, "weight", transform)

        # register output transformation hook
        elif args.location == TransformLocation.OUTPUT:

            def output_hook(_, _input, output):
                return transform(output)

            module.register_forward_hook(output_hook)

        # other locations such as q_attn and k_attn have not been implemented
        else:
            raise NotImplementedError()

        # (1) even in the `weight` cases, this submodule attachment is needed in order
        # to support saving in the frozen state


class TransformBase(Module, ABC):
    """
    Represents the application of a transform accord to TransformArgs
    """

    args: TransformArgs
    weight: Parameter

    @abstractmethod
    def forward(self, value: Tensor) -> Tensor:
        raise NotImplementedError()

    def right_inverse(self, value: Tensor) -> Tensor:
        with patch_attr(self.args, "inverse", not self.args.inverse):
            return self.forward(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(inverse={self.args.inverse})"
