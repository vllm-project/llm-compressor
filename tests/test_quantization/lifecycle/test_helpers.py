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
from compressed_tensors.utils.permute import safe_permute
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cpu"), torch.device("cuda"), torch.device("meta")]
)
def test_safe_permute(dtype: torch.dtype, device: torch.device):
    value = torch.tensor([[0, 1, 2, 3]], dtype=dtype, device=device)
    perm = torch.tensor([3, 1, 0, 2], device=device)

    result = safe_permute(value, perm, dim=-1)

    if device.type != "meta":
        assert torch.equal(result.squeeze(0), perm.to(result.dtype))
