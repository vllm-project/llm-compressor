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

import math
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open


REPO_PATH = Path(__file__).parent / "hadamards.safetensors"


__all__ = ["random_hadamard_matrix", "deterministic_hadamard_matrix", "is_pow2"]


# note that hadamard matrix multiplication can be accelerated using a library such as
# https://github.com/Dao-AILab/fast-hadamard-transform/tree/master


def deterministic_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Construct an n-by-n Hadamard matrix, using Sylvester's construction.
    `n` must be a power of 2.

    Adapated from https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_special_matrices.py  # noqa: E501

    :param size: order of the matrix, must be a power of 2
    :param dtype: data type of matrix
    :param device: device to construct matrix on
    :return: hadamard matrix of size `size`
    """
    if size <= 0:
        raise ValueError("Cannot construct deterministic hadamard of size <= 0")

    log2 = int(math.log2(size))
    if size != 2**log2:
        raise ValueError("Cannot construct deterministic hadamard of size != 2^n")

    H = torch.tensor([[1]], dtype=dtype, device=device)

    # Sylvester's construction
    for _ in range(log2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))

    return H


def random_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Produces a randomly generated Hadamard matrix. Differs from
    `deterministic_hadamard_matrix` in that this function supports non powers of 2
    and randomization using a seeded generator

    Adapated from https://github.com/facebookresearch/SpinQuant/blob/main/utils/hadamard_utils.py  # noqa: E501
    Known matrices were retrieved from N. J. A. Sloane's Library of Hadamard Matrices http://www.neilsloane.com/hadamard/  # noqa: E501

    :param size: The dimension of the hamadard matrix
    :param dtype: data type of matrix
    :param device: device to construct matrix on
    :param gen: Optional generator random values
    :return: randomly generated hadamard matrix
    """
    Q = torch.randint(low=0, high=2, size=(size,), generator=gen, dtype=dtype)  # cpu
    Q = Q.to(device=device)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return _matmul_hadU(Q)


def is_pow2(n: int) -> bool:
    """
    Check if a number is a power of 2

    :param n: number to check
    :return: True iff `n` is a power of 2
    """
    return n > 0 and (n & (n - 1) == 0)


def _fetch_hadamard_divisor(
    n: int,
    dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
    file_path: str = REPO_PATH,
) -> Optional[torch.Tensor]:
    """
    Fetch a known hadamard matrix from the given file path. The returned matrix will
    be of of size `k` such that `n / k` is a power of two. Return None if no such
    matrix exists.

    Note: This function reopens the safetensors file every time it is called.
    This is technically inefficient, but a very small runtime cost and simpler
    than forcing callers to manage the file open context

    :param n: size of known hadamard matrix
    :return: a known hadamard matrix of size `n` if one exists, else None
    """
    with safe_open(file_path, framework="pt", device=str(device)) as file:
        divisors = sorted((int(key) for key in file.keys()), reverse=True)
        for divisor in divisors:
            if n % divisor == 0 and is_pow2(n // divisor):
                return file.get_tensor(str(divisor)).to(dtype=dtype)

    return None


def _matmul_hadU(X: torch.Tensor) -> torch.Tensor:
    size = X.size(0)
    dtype = X.dtype
    device = X.device

    # Check if we have the determined hadamard matrix
    hadK = _fetch_hadamard_divisor(size, dtype, device=device)
    if hadK is None:
        raise ValueError(f"Cannot construct random hadamard matrix of size {size}")
    K = hadK.size(0)

    # Reshape diag matrix with randomized -1/+1
    input = X.clone().view(-1, size, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    assert input.shape[1] == K
    del output

    # Do not explicitly repeat - OOM
    # input = torch.bmm(
    #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    # Use bcast instead
    input = hadK.view(1, K, K).to(input) @ input

    # normalize
    return input.view(X.shape)
