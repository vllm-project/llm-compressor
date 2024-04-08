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
import shutil

import pytest
import torch
from safetensors.torch import save_file
from sparsetensors import BitmaskCompressor, BitmaskConfig, BitmaskTensor


@pytest.mark.parametrize(
    "shape,sparsity,dtype",
    [
        [(512, 1024), 0.5, torch.float32],
        [(830, 545), 0.8, torch.float32],
        [(342, 512), 0.3, torch.bfloat16],
        [(256, 700), 0.9, torch.float16],
    ],
)
def test_bitmask_sizes(shape, sparsity, dtype):
    test_tensor = torch.rand(shape, dtype=dtype)
    mask = (test_tensor.abs() < (1 - sparsity)).int()
    test_tensor *= mask
    dense_state_dict = {"dummy.weight": test_tensor}

    sparsity_config = BitmaskConfig()
    compressor = BitmaskCompressor(config=sparsity_config)
    sparse_state_dict = compressor.compress(dense_state_dict)

    # each dense tensor has 4 parameters for compression
    assert len(dense_state_dict) * 4 == len(sparse_state_dict)

    # bitmask should be 1 bit per dense element, rounded up to nearest int8
    sparse_shape = sparse_state_dict["dummy.weight.shape"]
    assert torch.all(torch.eq(sparse_shape, torch.tensor(shape)))
    bitmask_shape = sparse_state_dict["dummy.weight.bitmask"].shape
    assert bitmask_shape[0] == sparse_shape[0]
    assert bitmask_shape[1] == int(math.ceil(sparse_shape[1] / 8.0))

    # one value for each non-zero weight
    values_shape = sparse_state_dict["dummy.weight.compressed"].shape
    assert values_shape[0] == torch.sum(test_tensor != 0)
    row_offsets_shape = sparse_state_dict["dummy.weight.row_offsets"].shape
    assert row_offsets_shape[0] == test_tensor.shape[0]


@pytest.mark.parametrize(
    "shape,sparsity,dtype",
    [
        [(256, 512), 0.5, torch.float32],
        [(128, 280), 0.8, torch.float32],
        [(1024, 256), 0.3, torch.bfloat16],
        [(511, 350), 0.7, torch.float16],
    ],
)
def test_match(shape, sparsity, dtype):
    test_tensor1 = torch.rand(shape, dtype=dtype)
    mask = (test_tensor1.abs() < (1 - sparsity)).int()
    test_tensor1 *= mask

    test_tensor2 = torch.rand(shape, dtype=dtype)
    mask = (test_tensor2.abs() < (1 - sparsity)).int()
    test_tensor2 *= mask

    dense_state_dict = {"dummy.weight": test_tensor1, "dummy2.weight": test_tensor2}

    for key in dense_state_dict.keys():
        dense_tensor = dense_state_dict[key]
        sparse_tensor = BitmaskTensor.from_dense(dense_tensor)
        decompressed = sparse_tensor.decompress()
        assert decompressed.dtype == dense_tensor.dtype == dtype
        assert torch.equal(dense_tensor, decompressed)


@pytest.mark.parametrize(
    "sparsity,dtype",
    [
        [0.5, torch.float32],
        [0.8, torch.float32],
        [0.3, torch.bfloat16],
        [0.7, torch.float16],
    ],
)
def test_reload_match(sparsity, dtype, tmp_path):
    test_tensor1 = torch.rand((256, 512), dtype=dtype)
    mask = (test_tensor1.abs() < (1 - sparsity)).int()
    test_tensor1 *= mask

    test_tensor2 = torch.rand((360, 720), dtype=dtype)
    mask = (test_tensor2.abs() < (1 - sparsity)).int()
    test_tensor2 *= mask

    dense_state_dict = {"dummy.weight": test_tensor1, "dummy2.weight": test_tensor2}

    sparsity_config = BitmaskConfig()
    compressor = BitmaskCompressor(config=sparsity_config)

    sparse_state_dict = compressor.compress(dense_state_dict)
    save_file(sparse_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense = compressor.decompress(tmp_path)

    for key, reconstructed_tensor in reconstructed_dense:
        dense_tensor = dense_state_dict[key]
        assert dense_tensor.dtype == reconstructed_tensor.dtype == dtype
        assert torch.equal(dense_tensor, reconstructed_tensor)

    shutil.rmtree(tmp_path)
