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
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from torch.nn import Linear


def create_config(bit_depth, quant_type, input_symmetry, weight_symmetry):
    print(quant_type)
    weights = QuantizationArgs(
        num_bits=bit_depth, type=quant_type, symmetric=weight_symmetry
    )
    if input_symmetry is not None:
        inputs = QuantizationArgs(
            num_bits=bit_depth, type=quant_type, symmetric=input_symmetry
        )
    else:
        inputs = None

    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"], weights=weights, input_activations=inputs
        )
    }
    config = QuantizationConfig(
        config_groups=config_groups, quantization_status=QuantizationStatus.CALIBRATION
    )
    return config


@torch.no_grad
@pytest.mark.parametrize("bit_depth", [4, 8])
@pytest.mark.parametrize("quant_type", ["int"])
@pytest.mark.parametrize("input_symmetry", [True, False, None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
def test_bit_depths(bit_depth, quant_type, input_symmetry, weight_symmetry):
    model = Linear(64, 64)
    quant_config = create_config(bit_depth, quant_type, input_symmetry, weight_symmetry)
    apply_quantization_config(model, quant_config)

    min = -1 * int(2**bit_depth / 2)
    max = int(2**bit_depth / 2) - 1

    inputs = torch.randn(32, 64)
    model(inputs)
    if input_symmetry is not None:
        assert model.input_zero_point >= min
        assert model.input_zero_point <= max

        input_max = torch.max(inputs)
        input_min = torch.min(inputs)
        diff_from_max = abs(
            abs(model.input_scale * (max - model.input_zero_point)) - abs(input_max)
        )
        diff_from_min = abs(
            abs(model.input_scale * abs(min - model.input_zero_point)) - abs(input_min)
        )
        assert diff_from_max < model.input_scale or diff_from_min < model.input_scale

    assert model.weight_zero_point >= min
    assert model.weight_zero_point <= max

    weight_max = torch.max(model.weight)
    weight_min = torch.min(model.weight)
    diff_from_max = abs(
        abs(model.weight_scale * (max - model.weight_zero_point)) - abs(weight_max)
    )
    diff_from_min = abs(
        abs(model.weight_scale * abs(min - model.weight_zero_point)) - abs(weight_min)
    )
    assert diff_from_max < model.weight_scale or diff_from_min < model.weight_scale

    quantized_weight = fake_quantize(
        model.weight,
        model.weight_scale,
        model.weight_zero_point,
        model.quantization_scheme.weights,
    )
    assert not torch.any(quantized_weight < min).item()
    assert not torch.any(quantized_weight > max).item()


@torch.no_grad
@pytest.mark.parametrize("bit_depth", [8])
@pytest.mark.parametrize("quant_type", ["float"])
@pytest.mark.parametrize("input_symmetry", [True, False, None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
def test_fp8(bit_depth, quant_type, input_symmetry, weight_symmetry):
    model = Linear(64, 64)
    quant_config = create_config(bit_depth, quant_type, input_symmetry, weight_symmetry)
    apply_quantization_config(model, quant_config)

    dtype_info = torch.finfo(torch.float8_e4m3fn)
    min = dtype_info.min
    max = dtype_info.max

    inputs = torch.randn(32, 64)
    model(inputs)
    assert model.weight_zero_point.dtype == torch.float8_e4m3fn
    model.weight_zero_point.data = model.weight_zero_point.to(model.weight.dtype)
    if input_symmetry is not None:
        assert model.input_zero_point.dtype == torch.float8_e4m3fn
        model.input_zero_point.data = model.input_zero_point.to(model.weight.dtype)
        assert model.input_zero_point >= min
        assert model.input_zero_point <= max

        input_max = torch.max(inputs)
        input_min = torch.min(inputs)
        diff_from_max = abs(
            abs(model.input_scale * (max - model.input_zero_point)) - abs(input_max)
        )
        diff_from_min = abs(
            abs(model.input_scale * abs(min - model.input_zero_point)) - abs(input_min)
        )
        assert diff_from_max < model.input_scale or diff_from_min < model.input_scale

    assert model.weight_zero_point >= min
    assert model.weight_zero_point <= max

    weight_max = torch.max(model.weight)
    weight_min = torch.min(model.weight)
    diff_from_max = abs(
        abs(model.weight_scale * (max - model.weight_zero_point)) - abs(weight_max)
    )
    diff_from_min = abs(
        abs(model.weight_scale * abs(min - model.weight_zero_point)) - abs(weight_min)
    )
    assert diff_from_max < model.weight_scale or diff_from_min < model.weight_scale

    quantized_weight = fake_quantize(
        model.weight,
        model.weight_scale,
        model.weight_zero_point,
        model.quantization_scheme.weights,
    )
    assert not torch.any(quantized_weight < min).item()
    assert not torch.any(quantized_weight > max).item()
