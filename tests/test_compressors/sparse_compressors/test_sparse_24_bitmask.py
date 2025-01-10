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
from compressed_tensors import Sparse24BitMaskTensor
from compressed_tensors.quantization import FP8_DTYPE
from compressed_tensors.utils import combine_shards, shard_tensor
from tests.testing_utils import generate_pruned_semi_structured_mat, requires_gpu


@pytest.fixture
def dense_matrix_fixture():
    def _generate_dense_matrix(M, K, dtype):
        return generate_pruned_semi_structured_mat(M, K, dtype)

    return _generate_dense_matrix


@pytest.fixture
def shard_validation():
    def _validate_shard_shapes(sharded_values, sharded_bitmask, expected_shapes):
        for shard_values, shard_bitmask, expected_shape in zip(
            sharded_values, sharded_bitmask, expected_shapes
        ):
            assert (
                shard_values.shape == expected_shape["compressed"]
            ), f"Shape mismatch: {shard_values.shape} != {expected_shape['compressed']}"
            assert (
                shard_bitmask.shape == expected_shape["bitmask"]
            ), f"Shape mismatch: {shard_bitmask.shape} != {expected_shape['bitmask']}"

    return _validate_shard_shapes


def validate_compression(dense_matrix, decompressed_tensor):
    """Validate that the decompressed tensor matches the original dense matrix."""
    dense_matrix = dense_matrix.to(decompressed_tensor.device)
    assert dense_matrix.dtype == decompressed_tensor.dtype, "Dtype mismatch"
    assert dense_matrix.shape == decompressed_tensor.shape, "Shape mismatch"
    assert torch.equal(dense_matrix, decompressed_tensor), "Decompression failed"


@pytest.mark.parametrize("dtype", [torch.int8])
def test_bitmask_compress_decompress(dense_matrix_fixture, dtype):
    M, K = 1024, 1024
    dense_matrix = dense_matrix_fixture(M, K, dtype)

    bitmask_tensor = Sparse24BitMaskTensor.from_dense(
        dense_matrix, sparsity_structure="2:4"
    )
    decompressed_tensor = bitmask_tensor.decompress()

    validate_compression(dense_matrix, decompressed_tensor)


@pytest.mark.parametrize(
    "dtype, M, K, shard_sizes, shard_dim, expected_shapes",
    [
        (
            torch.int8,
            2560,
            2048,
            [2048, 256, 256],
            0,
            [
                {"compressed": (2048, 1024), "bitmask": (2048, 2048 // 8)},
                {"compressed": (256, 1024), "bitmask": (256, 2048 // 8)},
                {"compressed": (256, 1024), "bitmask": (256, 2048 // 8)},
            ],
        ),
        (
            torch.int8,
            2048,
            2048,
            [1024, 1024],
            1,
            [
                {"compressed": (2048, 512), "bitmask": (2048, 2048 // 8 // 2)},
                {"compressed": (2048, 512), "bitmask": (2048, 2048 // 8 // 2)},
            ],
        ),
    ],
)
def test_bitmask_compress_decompress_sharded(
    dense_matrix_fixture,
    shard_validation,
    dtype,
    M,
    K,
    shard_sizes,
    shard_dim,
    expected_shapes,
):
    dense_matrix = dense_matrix_fixture(M, K, dtype)

    bitmask_tensor = Sparse24BitMaskTensor.from_dense(dense_matrix)
    compressed_values = bitmask_tensor.compressed
    compressed_bitmask = bitmask_tensor.bitmask

    if shard_dim == 1:
        compressed_shard_sizes = [size // 2 for size in shard_sizes]
        bitmask_shard_sizes = [size // 8 for size in shard_sizes]
    else:
        compressed_shard_sizes = shard_sizes
        bitmask_shard_sizes = shard_sizes

    sharded_compressed_values = shard_tensor(
        compressed_values, compressed_shard_sizes, dim=shard_dim
    )
    sharded_compressed_bitmask = shard_tensor(
        compressed_bitmask, bitmask_shard_sizes, dim=shard_dim
    )

    shard_validation(
        sharded_compressed_values, sharded_compressed_bitmask, expected_shapes
    )

    decompressed_shards = [
        Sparse24BitMaskTensor(
            shape=(expected_shape["bitmask"][0], expected_shape["bitmask"][1] * 8),
            compressed=shard_values,
            bitmask=shard_bitmask,
        ).decompress()
        for shard_values, shard_bitmask, expected_shape in zip(
            sharded_compressed_values, sharded_compressed_bitmask, expected_shapes
        )
    ]

    decompressed_combined = combine_shards(decompressed_shards, dim=shard_dim)
    validate_compression(dense_matrix, decompressed_combined)


# GPU-Specific Tests for FP8_DTYPE
@pytest.mark.parametrize("dtype", [FP8_DTYPE])
@requires_gpu
def test_bitmask_compress_decompress_fp8(dense_matrix_fixture, dtype):
    test_bitmask_compress_decompress(dense_matrix_fixture, dtype)


@pytest.mark.parametrize(
    "dtype, M, K, shard_sizes, shard_dim, expected_shapes",
    [
        (
            FP8_DTYPE,
            2560,
            2048,
            [2048, 256, 256],
            0,
            [
                {"compressed": (2048, 1024), "bitmask": (2048, 2048 // 8)},
                {"compressed": (256, 1024), "bitmask": (256, 2048 // 8)},
                {"compressed": (256, 1024), "bitmask": (256, 2048 // 8)},
            ],
        ),
        (
            FP8_DTYPE,
            2048,
            2048,
            [1024, 1024],
            1,
            [
                {"compressed": (2048, 512), "bitmask": (2048, 2048 // 8 // 2)},
                {"compressed": (2048, 512), "bitmask": (2048, 2048 // 8 // 2)},
            ],
        ),
    ],
)
@requires_gpu
def test_bitmask_compress_decompress_sharded_fp8(
    dense_matrix_fixture,
    shard_validation,
    dtype,
    M,
    K,
    shard_sizes,
    shard_dim,
    expected_shapes,
):
    test_bitmask_compress_decompress_sharded(
        dense_matrix_fixture,
        shard_validation,
        dtype,
        M,
        K,
        shard_sizes,
        shard_dim,
        expected_shapes,
    )
