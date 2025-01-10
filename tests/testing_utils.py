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
# flake8: noqa
import unittest

import pytest


def compressed_tensors_config_available():
    try:
        from transformers.utils.quantization_config import (  # noqa: F401
            CompressedTensorsConfig,
        )

        return True
    except ImportError:
        return False


def accelerate_availabe():
    try:
        import accelerate  # noqa: F401

        return True

    except ImportError:
        return False


_is_compressed_tensors_config_available = compressed_tensors_config_available()
_is_accelerate_available = accelerate_availabe()


def requires_hf_quantizer():
    return pytest.mark.skipif(
        not _is_compressed_tensors_config_available,
        reason="requires transformers>=4.45 to support CompressedTensorsHfQuantizer",
    )


def requires_accelerate():
    return pytest.mark.skipif(
        not _is_accelerate_available,
        reason="requires accelerate",
    )


def get_random_mat(M, K, dtype) -> "torch.Tensor":
    """
    :param M: number of rows
    :param K: number of columns
    :param dtype: data type of the matrix
    :return: random matrix of shape (M, K) with non-zero values
    """
    import torch
    from compressed_tensors.quantization import FP8_DTYPE

    rand_tensor_dtype = dtype
    if dtype in [torch.int8, FP8_DTYPE]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda()
    mat = mat.masked_fill_(mat == 0, 1)
    return mat.to(dtype)


def generate_pruned_semi_structured_mat(M, K, dtype) -> "torch.Tensor":
    """
    :param M: number of rows
    :param K: number of columns
    :param dtype: data type of the matrix
    :return: random matrix of shape (M, K) with 2:4 sparsity pattern
    """
    import torch
    from compressed_tensors.quantization import FP8_DTYPE

    mask = torch.Tensor([0, 0, 1, 1]).tile((M, K // 4)).bool()
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, FP8_DTYPE]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype)
    mat = mat.masked_fill_(mat == 0, 1)
    if dtype == FP8_DTYPE:
        # some float8_e4m3fn operations are not supported on CPU
        mat = mat.cuda()
        mask = mask.cuda()
    mat = mat * mask
    return mat.to(dtype)


def induce_sparsity(tensor, sparsity_ratio) -> "torch.Tensor":
    """
    Makes a tensor sparse by zeroing out a given fraction
    of its smallest absolute values.

    :param: weight_tensor (torch.Tensor): The input weight tensor.
    :param: sparsity_ratio (float): Fraction of weights to be zeroed
        (0 <= sparsity_ratio <= 1).
    :returns: torch.Tensor: Sparse version of the input tensor.
    """
    import torch

    if not (0 <= sparsity_ratio <= 1):
        raise ValueError("Sparsity ratio must be between 0 and 1.")

    # Flatten the tensor and compute the threshold for sparsity
    flattened = tensor.view(-1)
    k = int(sparsity_ratio * flattened.numel())

    if k > 0:
        threshold = torch.topk(flattened.abs(), k, largest=False).values.max()
        sparse_tensor = torch.where(
            tensor.abs() > threshold, tensor, torch.zeros_like(tensor)
        )
    else:
        sparse_tensor = tensor

    return sparse_tensor


def is_gpu_available():
    """
    :return: True if a GPU is available, False otherwise
    """
    try:
        import torch  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False


def requires_gpu(test_case):
    return unittest.skipUnless(is_gpu_available(), "test requires GPU")(test_case)
