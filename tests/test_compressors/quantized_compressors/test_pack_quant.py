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
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import ActivationOrdering
from safetensors.torch import save_file
from torch.nn.modules import Linear, Sequential


def get_dummy_quant_config(
    num_bits=4, strategy=None, group_size=None, actorder=None, symmetric=True
):
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=num_bits,
                strategy=strategy,
                group_size=group_size,
                actorder=actorder,
                symmetric=symmetric,
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
        "dummy.weight_zero_point": torch.tensor(0, dtype=torch.int8),
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

    # pack-compressor only needs the number of bits from the quant-args to decompress
    # all other information is extracted from the compressed data directly
    names_to_scheme = {
        "dummy": QuantizationArgs(num_bits=num_bits),
        "dummy2": QuantizationArgs(num_bits=num_bits),
    }
    quant_config = get_dummy_quant_config(num_bits, symmetric=False)

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
    "strategy",
    {QuantizationStrategy.GROUP, QuantizationStrategy.CHANNEL},
)
def test_asymmetric_packed_support(strategy):
    shape = (1024, 1024)

    group_size = None
    if strategy == QuantizationStrategy.GROUP:
        group_size = 128

    if strategy == QuantizationStrategy.CHANNEL:
        expected_shape = (shape[0], 1)
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = shape[1] // group_size
        expected_shape = (shape[0], max(num_groups, 1))

    dense_state_dict = {
        "dummy.weight": torch.rand(shape),
        "dummy.weight_scale": torch.rand(expected_shape).to(torch.float32),
        "dummy.weight_zero_point": torch.rand(expected_shape).to(torch.int8),
    }

    quant_config = get_dummy_quant_config(
        strategy=strategy.value, symmetric=False, group_size=group_size
    )

    compressor = PackedQuantizationCompressor(config=quant_config)
    quantized_modules_to_args = {"dummy": quant_config.config_groups["group_1"].weights}
    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=quantized_modules_to_args
    )

    # compressed state_dict adds one entry for shape
    assert len(dense_state_dict) + 1 == len(compressed_state_dict)
    assert compressed_state_dict["dummy.weight_packed"].dtype == torch.int32
    assert compressed_state_dict["dummy.weight_zero_point"].dtype == torch.int32
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32

    # check weight compressed and packed
    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)  # round each row up to nearest int32
    assert compressed_state_dict["dummy.weight_packed"].shape == (
        expected_rows,
        expected_columns,
    )
    assert torch.equal(compressed_state_dict["dummy.weight_shape"], torch.tensor(shape))

    # check zp compressed and packed
    packed_size_zp = math.ceil(shape[0] / 8)
    zp_factor = group_size if strategy == QuantizationStrategy.GROUP else shape[-1]
    assert compressed_state_dict["dummy.weight_zero_point"].shape == (
        packed_size_zp,
        shape[-1] // zp_factor,
    )


@pytest.mark.parametrize(
    "actorder",
    [
        ActivationOrdering.GROUP,
        ActivationOrdering.WEIGHT,
        None,
    ],
)
def test_actorder_reload_match(actorder, tmp_path, mock_per_group_calibration):
    model = Sequential(OrderedDict([("dummy", Linear(512, 1024, bias=None))]))
    group_size = 128
    quant_config = get_dummy_quant_config(
        strategy="group", group_size=group_size, actorder=actorder
    )
    apply_quantization_config(model, quant_config)

    # run calibration
    model.quantization_status = QuantizationStatus.CALIBRATION
    mock_per_group_calibration(
        model.dummy, base_name="weight", value=model.dummy.weight, group_size=group_size
    )
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


@pytest.mark.parametrize(
    "num_bits,values,expected_values",
    [
        (
            4,
            torch.tensor([[1]]),
            torch.tensor([[9]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor([[1]]),
            torch.tensor([[129]], dtype=torch.int32),
        ),
        # 0000 0000 0000 0000 1100 1011 1010 1001
        (4, torch.tensor([[1, 2, 3, 4]]), torch.tensor([[52137]], dtype=torch.int32)),
        # 0111 0110 0101 0100 0011 0010 0001 0000
        (
            4,
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]]),
            torch.tensor([[1985229328]], dtype=torch.int32),
        ),
        # 10000100 10000011 10000010 10000001
        (
            8,
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[-2071756159]], dtype=torch.int32),
        ),
        # 00000011 00000010 00000001 00000000
        (
            8,
            torch.tensor([[-128, -127, -126, -125]]),
            torch.tensor([[50462976]], dtype=torch.int32),
        ),
        (
            4,
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]]),
            torch.tensor([[1985229328, 52137]], dtype=torch.int32),
        ),
        (
            4,
            torch.tensor(
                [
                    [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, -8, -8, -8, -8],
                    [1, 2, 3, 4, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1],
                ]
            ),
            torch.tensor([[1985229328, 52137], [52137, 1985229328]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor(
                [
                    [1, 2, 3, 4],
                    [-128, -127, -126, -125],
                ]
            ),
            torch.tensor([[-2071756159], [50462976]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor(
                [
                    [1, 2, 3, 4, -128, -127, -126, -125],
                    [-128, -127, -126, -125, 1, 2, 3, 4],
                ]
            ),
            torch.tensor(
                [[-2071756159, 50462976], [50462976, -2071756159]], dtype=torch.int32
            ),
        ),
    ],
)
def test_pack_to_int32(num_bits, values, expected_values):
    values = values.to(torch.int8)
    packed_values = pack_to_int32(values, num_bits)
    assert torch.equal(packed_values, expected_values)
    assert packed_values.dtype == expected_values.dtype


@pytest.mark.parametrize(
    "num_bits,values,expected_tensor",
    [
        (
            4,
            torch.tensor([[9]], dtype=torch.int32),
            torch.tensor([[1]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[129]], dtype=torch.int32),
            torch.tensor([[1]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[52137]], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[1985229328]], dtype=torch.int32),
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[-2071756159]], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[50462976]], dtype=torch.int32),
            torch.tensor([[-128, -127, -126, -125]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[1985229328, 52137]], dtype=torch.int32),
            torch.tensor(
                [[-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]], dtype=torch.int8
            ),
        ),
        (
            4,
            torch.tensor([[1985229328, 52137], [52137, 1985229328]], dtype=torch.int32),
            torch.tensor(
                [
                    [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, -8, -8, -8, -8],
                    [1, 2, 3, 4, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1],
                ],
                dtype=torch.int8,
            ),
        ),
        (
            8,
            torch.tensor([[-2071756159], [50462976]], dtype=torch.int32),
            torch.tensor(
                [
                    [1, 2, 3, 4],
                    [-128, -127, -126, -125],
                ],
                dtype=torch.int8,
            ),
        ),
        (
            8,
            torch.tensor(
                [[-2071756159, 50462976], [50462976, -2071756159]], dtype=torch.int32
            ),
            torch.tensor(
                [
                    [1, 2, 3, 4, -128, -127, -126, -125],
                    [-128, -127, -126, -125, 1, 2, 3, 4],
                ],
                dtype=torch.int8,
            ),
        ),
    ],
)
def test_unpack_from_int32(num_bits, values, expected_tensor):
    unpacked_tensor = unpack_from_int32(values, num_bits, expected_tensor.shape)
    assert torch.equal(unpacked_tensor, unpacked_tensor)
    assert unpacked_tensor.dtype == unpacked_tensor.dtype
