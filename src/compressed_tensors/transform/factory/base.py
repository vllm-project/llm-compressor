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
from collections import defaultdict
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.utils.parametrize as P
import tqdm
from compressed_tensors.registry.registry import RegistryMixin, T
from compressed_tensors.transform import (
    TransformArgs,
    TransformLocation,
    TransformScheme,
)
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_module,
    has_offloaded_params,
    match_named_modules,
    patch_attr,
    register_offload_module,
    update_offload_parameter,
)
from compressed_tensors.utils.internal import InternalModule
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

    transforms: List["TransformBase"]

    def __init__(self, name: str, scheme: TransformScheme, seed: Optional[int] = None):
        self.name = name
        self.scheme = scheme
        self.generator = torch.Generator()
        self.transforms = list()
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

    def apply_to_model(self, model: Module, use_tqdm=True):
        """
        Create transforms and apply them to the model

        :param model: module to apply transforms to
        """
        modules_args = [
            (module, arg)
            for arg in self.scheme.apply
            for _, module in match_named_modules(model, arg.targets, arg.ignore)
        ]

        desc = f"Applying {self.name} transforms"
        for module, arg in tqdm.tqdm(modules_args, desc=desc, disable=(not use_tqdm)):
            self._apply_to_module(module, arg)

        self._update_tied_weights()

    def _apply_to_module(self, module: Module, args: TransformArgs):
        """
        Create transforms and apply them to the module

        :param module: target module to apply transforms to
        :param args: defines how the transform will be applied to the target module
        """
        if has_offloaded_params(module):
            if module._hf_hook.place_submodules:
                raise NotImplementedError(
                    "Applying transforms to offloaded submodules with "
                    "`place_submodules=True` is not supported"
                )

        # create transform as submodule
        transform_name = f"{self.name}_{args.location}"
        transform = self.create_transform(module, args)
        self.transforms.append(transform)
        register_offload_module(module, transform_name, transform)

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
            # fuse transform into weight
            assert hasattr(module, "weight")
            with torch.no_grad(), align_module_device(module):
                update_offload_parameter(module, "weight", transform(module.weight))

            if self.scheme.requires_grad:
                # for training, the weight changes with every forward pass
                # so we can leverage parametrization to propagate the gradient
                if has_offloaded_params(module):
                    raise ValueError("Offloaded training is not supported")
                P.register_parametrization(module, "weight", transform)

            else:
                # transform is no longer needed (unfusing is not supported)
                delete_offload_module(module, transform_name)

        # register output transformation hook
        elif args.location == TransformLocation.OUTPUT:

            def output_hook(_, _input, output):
                return transform(output)

            module.register_forward_hook(output_hook)

        # other locations such as q_attn and k_attn have not been implemented
        else:
            raise NotImplementedError()

    def _update_tied_weights(self):
        """
        Populate the `_dynamic_tied_weights_keys` attribute of transforms,
        which is used by transformers to detect and remove shared pointers
        during saving
        """
        # map from data_ptrs to keys
        ptr_to_keys: dict[int, List[Tuple[TransformBase, str]]] = defaultdict(list)
        for transform in self.transforms:
            for name, param in transform.named_parameters(recurse=False):
                # NOTE: previously asserted that parent._hf_hook.place_submodules=False
                if has_offloaded_params(transform):
                    param = transform._hf_hook.weights_map[name]
                ptr_to_keys[param.data_ptr()].append((transform, name))

        # populate `_dynamic_tied_weights_keys` if there is more than one key
        # and ensure that they share tensors
        for shared_keys in ptr_to_keys.values():
            if len(shared_keys) > 1:
                tensor = getattr(shared_keys[0][0], shared_keys[0][1])

                for transform, name in shared_keys:
                    transform._dynamic_tied_weights_keys.add(name)
                    setattr(transform, name, tensor)


class TransformBase(InternalModule, ABC):
    """
    Represents the application of a transform accord to TransformArgs
    """

    args: TransformArgs
    weight: Parameter
    _dynamic_tied_weights_keys: Set[str]

    def __init__(self):
        super().__init__()
        self._dynamic_tied_weights_keys = set()

    @abstractmethod
    def forward(self, value: Tensor) -> Tensor:
        raise NotImplementedError()

    def right_inverse(self, value: Tensor) -> Tensor:
        with patch_attr(self.args, "inverse", not self.args.inverse):
            return self.forward(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(inverse={self.args.inverse})"
