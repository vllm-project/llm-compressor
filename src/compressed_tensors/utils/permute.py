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

from typing import Set, Tuple

import torch


__all__ = ["safe_permute"]


# these datatypes are missing implementations required for standard permutation
_EXPERIMENTAL_DTYPES: Set[Tuple[torch.dtype, torch.device]] = set()


def safe_permute(value: torch.Tensor, perm: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Perform out-of-place permutation without using torch.Tensor.index_put_,
    whose implementation is missing for datatypes such as `torch.float8_e4m3fn`

    :param value: tensor to permute
    :param perm: permutation map
    :param dim: dimension along which to apply permutation
    :return: permuted value
    """
    dtype_tuple = (value.dtype, value.device)

    if dtype_tuple in _EXPERIMENTAL_DTYPES:
        return _fallback_permute(value, perm, dim)

    try:
        return value[tuple([slice(None)] * dim + [perm])]
    except RuntimeError:
        # Mark dtype as experimental if advanced indexing fails
        _EXPERIMENTAL_DTYPES.add(dtype_tuple)
        return _fallback_permute(value, perm, dim)


def _fallback_permute(
    value: torch.Tensor, perm: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Fallback permutation method for experimental dtypes.

    :param value: tensor to permute
    :param perm: permutation map
    :param dim: dimension along which to apply permutation
    :return: permuted value
    """
    value_ret = value.clone()  # cannot use zeros_like b/c of missing impl.
    orig_slices = [slice(None)] * (dim + 1)
    perm_slices = [slice(None)] * (dim + 1)

    for index, perm_index in enumerate(perm):
        orig_slices[dim] = index
        perm_slices[dim] = perm_index
        value_ret[tuple(orig_slices)] = value[tuple(perm_slices)]

    return value_ret
