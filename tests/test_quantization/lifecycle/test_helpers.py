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


import pytest
import torch
from compressed_tensors.utils import safe_permute
from compressed_tensors.utils.permute import _EXPERIMENTAL_DTYPES


@pytest.mark.parametrize(
    "dtype,device,exp_experimental",
    [
        (torch.int8, torch.device("cpu"), False),
        (torch.int16, torch.device("cpu"), False),
        (torch.int32, torch.device("cpu"), False),
        (torch.int64, torch.device("cpu"), False),
        (torch.float16, torch.device("cpu"), False),
        (torch.float32, torch.device("cpu"), False),
        (torch.float64, torch.device("cpu"), False),
        (torch.float8_e4m3fn, torch.device("cpu"), True),
    ],
)
def test_safe_permute(dtype: torch.dtype, device: str, exp_experimental: bool):
    # some dtypes do not support arange initialization
    tensor = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
    perm = torch.tensor([3, 1, 0, 2])
    expected = torch.tensor([3, 1, 0, 2], dtype=dtype, device=device)

    result = safe_permute(tensor, perm, dim=0)

    if exp_experimental:
        assert (dtype, device) in _EXPERIMENTAL_DTYPES
    assert all(result == expected)
