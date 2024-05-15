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
from compressed_tensors import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized import (
    pack_4bit_ints,
    unpack_4bit_ints,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from safetensors.torch import save_file


def get_dummy_quant_config():
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"], weights=QuantizationArgs(num_bits=4)
        ),
    }
    ignore = ["lm_head"]
    quant_config = QuantizationConfig(
        config_groups=config_groups,
        ignore=ignore,
    )

    return quant_config


@pytest.mark.parametrize(
    "shape",
    [
        (512, 1024),
        (830, 545),
        (342, 512),
        (256, 700),
    ],
)
def test_quant_format(shape):
    dense_state_dict = {
        "dummy.weight": torch.rand(shape),
        "dummy.weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(0, dtype=torch.int32),
    }
    quant_config = get_dummy_quant_config()

    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {"dummy": quant_config.config_groups["group_1"].weights}
    compressed_state_dict = compressor.compress(
        dense_state_dict, model_quant_args=quantized_modules_to_args
    )

    # compressed state_dict adds one entry for shape
    assert len(dense_state_dict) + 1 == len(compressed_state_dict)

    # check compressed and packed
    assert compressed_state_dict["dummy.weight"].dtype == torch.int32
    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)  # round each row up to nearest int32
    assert compressed_state_dict["dummy.weight"].shape == (
        expected_rows,
        expected_columns,
    )

    assert torch.equal(compressed_state_dict["dummy.weight_shape"], torch.tensor(shape))
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32
    assert compressed_state_dict["dummy.weight_zero_point"].dtype == torch.int32


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [-1, -2, -3, -4, -5, -6, -7, -8]]),
        (torch.rand((32, 100)) * 16 - 8),
    ],
)
def test_repack(value):
    value = value.to(torch.int8)
    shape = value.shape
    assert not torch.any(value > 7).item()
    assert not torch.any(value < -8).item()

    packed = pack_4bit_ints(value)
    unpacked = unpack_4bit_ints(packed, shape)
    assert torch.equal(value, unpacked)


def test_reload_match(tmp_path):
    dense_state_dict = {
        "dummy.weight": torch.rand((511, 350)),
        "dummy.weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(0, dtype=torch.int32),
        "dummy2.weight": torch.rand((128, 280)),
        "dummy2.weight_scale": torch.tensor(0.02, dtype=torch.float32),
        "dummy2.weight_zero_point": torch.tensor(15, dtype=torch.int32),
    }
    quant_config = get_dummy_quant_config()

    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {
        "dummy": quant_config.config_groups["group_1"].weights,
        "dummy2": quant_config.config_groups["group_1"].weights,
    }
    compressed_state_dict = compressor.compress(
        dense_state_dict, model_quant_args=quantized_modules_to_args
    )
    save_file(compressed_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense_gen = compressor.decompress(tmp_path)
    reconstructed_dense = {}
    for name, value in reconstructed_dense_gen:
        reconstructed_dense[name] = value

    fake_quant_dummy = fake_quantize(
        dense_state_dict["dummy.weight"],
        scale=dense_state_dict["dummy.weight_scale"],
        zero_point=dense_state_dict["dummy.weight_zero_point"],
        args=quantized_modules_to_args["dummy"],
    )
    assert torch.equal(
        fake_quant_dummy, reconstructed_dense["dummy.weight"].to(torch.float32)
    )

    fake_quant_dummy2 = fake_quantize(
        dense_state_dict["dummy2.weight"],
        scale=dense_state_dict["dummy2.weight_scale"],
        zero_point=dense_state_dict["dummy2.weight_zero_point"],
        args=quantized_modules_to_args["dummy2"],
    )
    assert torch.equal(
        fake_quant_dummy2, reconstructed_dense["dummy2.weight"].to(torch.float32)
    )

    shutil.rmtree(tmp_path)
