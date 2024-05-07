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

from copy import deepcopy

import torch
from compressed_tensors.quantization.lifecycle.calibration import (
    set_module_for_calibration,
)
from compressed_tensors.quantization.lifecycle.frozen import freeze_module_quantization
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Linear


def test_lifecyle(create_quantization_scheme):
    num_bits = 8

    quantization_scheme = create_quantization_scheme(
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        targets=["*"],
    )

    layer = Linear(4, 4)
    layer.weight.data *= 100

    # updated layer keys check
    expected_layer_keys = {"weight", "bias"}
    for key in layer.state_dict().keys():
        expected_layer_keys.remove(key)
    assert len(expected_layer_keys) == 0

    # over write forward pass and register zero_point and scale
    initialize_module_for_quantization(layer, quantization_scheme)
    expected_layer_keys = {
        "input_scale",
        "input_zero_point",
        "weight_scale",
        "weight_zero_point",
        "weight",
        "bias",
    }
    for key in layer.state_dict().keys():
        expected_layer_keys.remove(key)
    assert len(expected_layer_keys) == 0

    # should have both input and weight observer after initalizing
    assert hasattr(layer, "input_observer")
    assert hasattr(layer, "weight_observer")

    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")
    assert layer.quantization_status == QuantizationStatus.INITIALIZED

    set_module_for_calibration(layer)
    assert layer.quantization_status == QuantizationStatus.CALIBRATION

    # do a calibration step
    assert torch.numel(layer.input_zero_point.data) == 0
    assert torch.numel(layer.input_scale) == 0
    assert torch.numel(layer.weight_scale) == 0
    assert torch.numel(layer.weight_zero_point) == 0

    layer(torch.randn(4, 4))

    # zero-points and scale should be updated after forward pass
    assert torch.numel(layer.input_zero_point.data) > 0
    assert torch.numel(layer.input_scale) > 0
    assert torch.numel(layer.weight_scale) > 0
    assert torch.numel(layer.weight_zero_point) > 0

    # symmetric zero points should center at 0
    assert layer.weight_zero_point.data == 0

    # check high and low bound of the weights
    assert torch.all(layer.weight.data >= -128) and torch.all(layer.weight.data <= 127)

    initialized_layer_input_zero_point = deepcopy(layer.input_zero_point)
    initialized_layer_input_scale = deepcopy(layer.input_scale)
    initialized_layer_weight_scale = deepcopy(layer.weight_scale)
    # calibrate the layers with each iteration
    for _ in range(10):
        layer(torch.randn(4, 4))

    assert initialized_layer_input_zero_point != 0
    assert initialized_layer_input_scale != layer.input_scale
    assert initialized_layer_weight_scale == layer.weight_scale

    # check quantization f_q(x) is applied after frozen without update
    input_check_for_quant = torch.randn(4, 4)
    out_calibration = layer(input_check_for_quant)

    layer_before_freeze_input_zero_point = deepcopy(layer.input_zero_point)
    layer_before_freeze_input_scale = deepcopy(layer.input_scale)
    layer_before_freeze_weight_scale = deepcopy(layer.weight_scale)

    # Freeze, no update after any forward pass
    freeze_module_quantization(layer)

    for _ in range(10):
        layer(torch.randn(4, 4))
    assert layer_before_freeze_input_zero_point == layer.input_zero_point
    assert layer_before_freeze_input_scale == layer.input_scale
    assert layer_before_freeze_weight_scale == layer.weight_scale

    # check that the same quantization is applied as calibration to frozen
    assert torch.all(out_calibration == layer(input_check_for_quant))
