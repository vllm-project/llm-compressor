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
from collections import OrderedDict

import pytest
import torch
from compressed_tensors import PackedQuantizationCompressor
from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    apply_quantization_status,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import ActivationOrdering
from safetensors.torch import save_file
from torch.nn.modules import Linear, Sequential


def get_dummy_quant_config(num_bits=4, strategy=None, group_size=None, actorder=None):
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=num_bits,
                strategy=strategy,
                group_size=group_size,
                actorder=actorder,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.nn.Parameter(
        (torch.arange(columns, dtype=torch.int) // group_size)[perm],
        requires_grad=False,
    )


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
        dense_state_dict, names_to_scheme=quantized_modules_to_args
    )

    # compressed state_dict adds one entry for shape
    # but removes the zero points since we are symmetric
    assert len(dense_state_dict) == len(compressed_state_dict)

    # check compressed and packed
    assert compressed_state_dict["dummy.weight_packed"].dtype == torch.int32
    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)  # round each row up to nearest int32
    assert compressed_state_dict["dummy.weight_packed"].shape == (
        expected_rows,
        expected_columns,
    )

    assert torch.equal(compressed_state_dict["dummy.weight_shape"], torch.tensor(shape))
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [-1, -2, -3, -4, -5, -6, -7, -8]]),
        (torch.rand((32, 100)) * 16 - 8),
    ],
)
def test_repack_4bit(value):
    value = value.to(torch.int8)
    shape = value.shape
    assert not torch.any(value > 7).item()
    assert not torch.any(value < -8).item()

    packed = pack_to_int32(value, 4)
    unpacked = unpack_from_int32(packed, 4, shape)
    assert torch.equal(value, unpacked)


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[30, 40], [50, 60]]),
        torch.tensor(
            [[10, 15, 20, 25, 30, 35, 40, 45], [-10, -20, -30, -40, -50, -60, -70, -80]]
        ),
        (torch.rand((32, 100)) * 256 - 128),
    ],
)
def test_repack_8bit(value):
    value = value.to(torch.int8)
    shape = value.shape
    assert not torch.any(value > 127).item()
    assert not torch.any(value < -128).item()

    packed = pack_to_int32(value, 8)
    unpacked = unpack_from_int32(packed, 8, shape)
    assert torch.equal(value, unpacked)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_reload_match(tmp_path, num_bits):
    dense_state_dict = {
        "dummy.weight": torch.rand((511, 350)),
        "dummy.weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(0, dtype=torch.int8),
        "dummy2.weight": torch.rand((128, 280)),
        "dummy2.weight_scale": torch.tensor(0.02, dtype=torch.float32),
        "dummy2.weight_zero_point": torch.tensor(15, dtype=torch.int8),
    }

    names_to_scheme = {
        "dummy": QuantizationArgs(num_bits=num_bits),
        "dummy2": QuantizationArgs(num_bits=num_bits),
    }
    quant_config = get_dummy_quant_config(num_bits)

    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {
        "dummy": quant_config.config_groups["group_1"].weights,
        "dummy2": quant_config.config_groups["group_1"].weights,
    }
    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=quantized_modules_to_args
    )
    save_file(compressed_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense_gen = compressor.decompress(
        tmp_path, names_to_scheme=names_to_scheme
    )
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


@pytest.mark.parametrize(
    "actorder",
    [
        ActivationOrdering.GROUP,
        ActivationOrdering.WEIGHT,
        None,
    ],
)
def test_actorder_reload_match(actorder, tmp_path):
    model = Sequential(OrderedDict([("dummy", Linear(512, 1024, bias=None))]))
    group_size = 128
    quant_config = get_dummy_quant_config(
        strategy="group", group_size=group_size, actorder=actorder
    )
    apply_quantization_config(model, quant_config)

    # run calibration
    apply_quantization_status(model, QuantizationStatus.CALIBRATION)
    for _ in range(16):
        inputs = torch.rand((512, 512))
        _ = model(inputs)
    apply_quantization_status(model, QuantizationStatus.FROZEN)

    # apply gptq
    if actorder == ActivationOrdering.GROUP:
        init_g_idx = make_dummy_g_idx(512, group_size)
        model.dummy.register_parameter("weight_g_idx", init_g_idx)

    # compress
    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {
        "dummy": quant_config.config_groups["group_1"].weights,
    }
    compressed_state_dict = compressor.compress(
        model.state_dict(), names_to_scheme=quantized_modules_to_args
    )
    save_file(compressed_state_dict, tmp_path / "model.safetensors")

    # decompress
    reconstructed_dense_gen = compressor.decompress(
        tmp_path, names_to_scheme=quantized_modules_to_args
    )
    reconstructed_dense = {}
    for name, value in reconstructed_dense_gen:
        reconstructed_dense[name] = value

    fake_quant_dummy = fake_quantize(
        model.dummy.weight,
        scale=model.dummy.weight_scale,
        zero_point=model.dummy.weight_zero_point,
        g_idx=getattr(model.dummy, "weight_g_idx", None),
        args=quantized_modules_to_args["dummy"],
    )
    assert torch.equal(fake_quant_dummy, reconstructed_dense["dummy.weight"])

    shutil.rmtree(tmp_path)
