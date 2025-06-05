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


import numpy
import pytest
import torch
from compressed_tensors.transform.utils.hadamard import (
    _get_had12,
    _get_had20,
    deterministic_hadamard_matrix,
    random_hadamard_matrix,
)


@pytest.mark.parametrize(
    "had_func",
    [
        _get_had12,
        _get_had20,
    ],
)
def test_packed_hadamard_compliant(had_func):
    had_matrix = had_func()
    size = had_matrix.size(0)
    # HH.T == nI
    product = had_matrix @ had_matrix.T
    assert torch.equal(product, size * torch.eye(size))


@pytest.mark.parametrize(
    "size",
    [4096, 2048],
)
def test_random_hadamard_matrix_compliant(size):
    had_matrix = random_hadamard_matrix(size)
    product = torch.round(had_matrix @ had_matrix.T)
    assert torch.equal(product, torch.eye(size))


def test_random_hadamard_generator():
    generator = torch.Generator().manual_seed(42)
    one = random_hadamard_matrix(2048, generator)
    two = random_hadamard_matrix(2048, generator)

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


@pytest.mark.parametrize(
    "size",
    [1024],
)
def test_deterministic_hadamard_compliant(size):
    had_matrix = deterministic_hadamard_matrix(size)
    # (H / sqrt(n))(H.T / sqrt(n)) == I
    product = had_matrix @ had_matrix.T
    assert numpy.array_equal(product, numpy.eye(size))
