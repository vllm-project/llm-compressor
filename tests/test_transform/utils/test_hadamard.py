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
from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix,
    is_pow2,
    random_hadamard_matrix,
)
from tests.testing_utils import requires_gpu


_sizes_to_test = [
    768,  # gpt2 small
    1024,  # gpt2 medium
    1280,  # qwen_2_5_vl vision
    1600,  # gpt2 xl
    2048,  # gpt3 small
    3584,  # qwen_2_5_vl
    3840,  # qwen_2_5_vl vision qkv
    4096,  # llama3
    7168,  # deepseek_v3
    14336,  # llama3 intermediate
    18432,  # deepseek_v3 intermediate
    18944,  # qwen_2_5_vl intermediate
]
_atol = 1e-1  # bfloat16 is low precision for large matrices


@requires_gpu
@pytest.mark.parametrize("size", _sizes_to_test)
def test_random_hadamard_matrix_compliant(size):
    # (H / sqrt(n))(H.T / sqrt(n)) == I
    matrix = random_hadamard_matrix(size, device="cuda")
    product = matrix @ matrix.T
    eye = torch.eye(size, dtype=product.dtype, device="cuda")
    assert torch.allclose(product, eye, atol=_atol)


def test_random_hadamard_generator():
    # check that generation is deterministic with a seed
    generator = torch.Generator().manual_seed(42)
    one = random_hadamard_matrix(2048, gen=generator)
    two = random_hadamard_matrix(2048, gen=generator)

    one_true = torch.tensor(
        [
            [-1, -1, -1],
            [+1, -1, +1],
            [-1, -1, +1],
        ]
    )
    two_true = torch.tensor(
        [
            [-1, -1, -1],
            [-1, +1, -1],
            [+1, +1, -1],
        ]
    )

    assert torch.all(one[:3, :3].sign() == one_true.sign())
    assert torch.all(two[:3, :3].sign() == two_true.sign())


@requires_gpu
@pytest.mark.parametrize("size", _sizes_to_test)
def test_deterministic_hadamard_compliant(size):
    if not is_pow2(size):
        with pytest.raises(ValueError):
            matrix = deterministic_hadamard_matrix(size, device="cuda")
        return

    # (H / sqrt(n))(H.T / sqrt(n)) == I
    matrix = deterministic_hadamard_matrix(size, device="cuda")
    product = matrix @ matrix.T
    eye = torch.eye(size, dtype=product.dtype, device="cuda")
    assert torch.allclose(product, eye, atol=_atol)
