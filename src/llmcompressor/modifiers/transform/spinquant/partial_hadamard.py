# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.transform import TransformArgs, TransformLocation, TransformScheme
from compressed_tensors.transform.factory.base import TransformFactory
from compressed_tensors.transform.factory.hadamard import (
    HadamardFactory,
    HadamardTransform,
)
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from compressed_tensors.transform.utils.matrix import (
    _multihead_matmul,
    get_transform_size,
)
from compressed_tensors.utils import get_execution_device, get_offloaded_device
from compressed_tensors.utils.helpers import ParameterizedDefaultDict
from torch import Tensor, device, dtype
from torch.nn import Module, Parameter


@TransformFactory.register("partial_hadamard")
class PartialHadamardFactory(HadamardFactory):
    """
    Factory for partial hadamard transforms, used for models (e.g. DeepseekV3)
    where KV weight is combined and only the V rows need to be transformed.

    Set ``qk_nope_head_dim`` and ``v_head_dim`` on this class before calling
    ``apply_transform_config`` so the transforms know the chunk layout.

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    qk_nope_head_dim: int = 0
    v_head_dim: int = 0

    def __init__(self, name: str, scheme: TransformScheme, seed: int | None = None):
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
        return PartialHadamardTransform(
            weight, perm, self.scheme, args, type(module),
            qk_nope_head_dim=PartialHadamardFactory.qk_nope_head_dim,
            v_head_dim=PartialHadamardFactory.v_head_dim,
        )

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


class PartialHadamardTransform(HadamardTransform):
    def __init__(
        self,
        weight: Parameter,
        perm: Parameter | None,
        scheme: TransformScheme,
        args: TransformArgs,
        module_type: type[torch.nn.Module],
        qk_nope_head_dim: int = 0,
        v_head_dim: int = 0,
    ):
        super().__init__(weight, perm, scheme, args, module_type)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight

        if self.perm is not None:
            weight = weight[self.perm][:, self.perm]

        if self.args.inverse:
            weight = weight.T

        result = apply_partial_transform_weight(
            weight.to(device=value.device),
            value.to(dtype=weight.dtype),
            self.args.location,
            self.module_type,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        # Chunked WEIGHT_OUTPUT path scales only V rows inside; do not scale again.
        if self.args.location == TransformLocation.WEIGHT_OUTPUT:
            return result.to(value.dtype)
        return (result / self._scale).to(value.dtype)


def apply_partial_transform_weight(
    transform_weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
    module_type: type[torch.nn.Module],
    qk_nope_head_dim: int = 0,
    v_head_dim: int = 0,
) -> torch.Tensor:
    """
    Apply the transform_weight to the given value, but only to the V rows
    of each KV chunk (skipping the qk_nope rows).

    For ``WEIGHT_OUTPUT``, the value has shape
    ``[num_kv_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]``.
    Only the ``v_head_dim`` rows in each chunk are multiplied by the
    Hadamard matrix; the ``qk_nope_head_dim`` rows are left unchanged.

    :param transform_weight: square Hadamard matrix
    :param value: weight tensor to transform
    :param location: determines how weight should be applied
    :param module_type: result of type(module)
    :param qk_nope_head_dim: number of qk_nope rows per KV head to skip
    :param v_head_dim: number of V rows per KV head to transform
    :return: value after transform_weight has been applied
    """
    assert transform_weight.shape[0] == transform_weight.shape[1]
    assert qk_nope_head_dim > 0 and v_head_dim > 0
    if TransformLocation(location).is_online():
        return _multihead_matmul(value, transform_weight)

    if module_type == torch.nn.Linear:
        if location == TransformLocation.WEIGHT_INPUT:
            return _multihead_matmul(value, transform_weight.T)
        elif location == TransformLocation.WEIGHT_OUTPUT:
            chunk_size = qk_nope_head_dim + v_head_dim
            num_chunks = value.shape[0] // chunk_size
            result = value.clone()

            scale = torch.tensor(v_head_dim, dtype=torch.float64).sqrt()
            for i in range(num_chunks):
                start_idx = i * chunk_size + qk_nope_head_dim
                end_idx = (i + 1) * chunk_size
                result[start_idx:end_idx, :] = (
                    _multihead_matmul(transform_weight.T, value[start_idx:end_idx, :])
                    / scale
                )
            return result

    raise NotImplementedError(
        f"Applying transforms to {module_type} {location} is not supported"
    )