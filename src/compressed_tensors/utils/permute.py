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

import torch
from compressed_tensors.utils.helpers import deprecated


__all__ = ["safe_permute"]


@deprecated("Tensor.index_select")
def safe_permute(value: torch.Tensor, perm: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Perform out-of-place permutation without using torch.Tensor.index_put_,
    whose implementation is missing for datatypes such as `torch.float8_e4m3fn`

    :param value: tensor to permute
    :param perm: permutation map
    :param dim: dimension along which to apply permutation
    :return: permuted value
    """
    return value.index_select(dim, perm)
