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
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from torch.nn import Linear


def create_config(input_symmetry, weight_symmetry, w_strategy, i_strategy=None, group_size=None):
    weights = QuantizationArgs(
        symmetric=weight_symmetry, strategy=w_strategy, group_size=group_size
    )
    if input_symmetry is not None:
        inputs = QuantizationArgs(
            symmetric=input_symmetry, strategy=i_strategy, group_size=group_size
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
@pytest.mark.parametrize("input_symmetry", [None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
@pytest.mark.parametrize("model_shape", [(64,128), (300, 200), (400,400)])
def test_channelwise(input_symmetry, weight_symmetry, model_shape):
    model = Linear(model_shape[0], model_shape[1])
    quant_config = create_config(
        input_symmetry, weight_symmetry, w_strategy=QuantizationStrategy.CHANNEL
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(32, model_shape[0])
    model(inputs)

    assert list(model.weight_scale.shape) == [model_shape[1], 1]
    assert list(model.weight_zero_point.shape) == [model_shape[1], 1]

@torch.no_grad
@pytest.mark.parametrize("input_symmetry", [None])
@pytest.mark.parametrize("weight_symmetry", [True, False])
@pytest.mark.parametrize("model_shape", [(128,256), (256, 512), (512,1024)])
@pytest.mark.parametrize("group_size", [32,128])
def test_group(input_symmetry, weight_symmetry, model_shape, group_size):
    model = Linear(model_shape[0], model_shape[1])
    quant_config = create_config(
        input_symmetry, weight_symmetry, w_strategy=QuantizationStrategy.GROUP, group_size=group_size
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(128, model_shape[0])
    model(inputs)

    assert list(model.weight_scale.shape) == [model_shape[1], int(model_shape[0] / group_size), 1]
    assert list(model.weight_zero_point.shape) == [model_shape[1], int(model_shape[0] / group_size), 1]


@torch.no_grad
@pytest.mark.parametrize("input_symmetry", [True, False])
@pytest.mark.parametrize("weight_symmetry", [True, False])
@pytest.mark.parametrize("input_shape", [(32,256), (300, 200), (400,400)])
def test_token(input_symmetry, weight_symmetry, input_shape):
    model = Linear(input_shape[1], 256)
    quant_config = create_config(
        input_symmetry, weight_symmetry, w_strategy=QuantizationStrategy.CHANNEL, i_strategy=QuantizationStrategy.TOKEN
    )
    apply_quantization_config(model, quant_config)

    inputs = torch.randn(input_shape)
    model(inputs)

    assert list(model.input_scale.shape) == [1,input_shape[1]]
    assert list(model.input_zero_point.shape) == [1,input_shape[1]]

    assert list(model.weight_scale.shape) == [256, 1]
    assert list(model.weight_zero_point.shape) == [256, 1]