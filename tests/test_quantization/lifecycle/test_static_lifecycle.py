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
    QuantizationScheme,
    forward_quantize,
    initialize_module_for_quantization,
    initialize_qparams,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from tests.mock_observer import MockMinMaxObserver


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_quant,exp_loss",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",  # equivalent to token
            ),
            torch.tensor([0.0]),
            torch.tensor([23.0]),
            torch.tensor(
                [
                    [0.0000, 0.0000, 3.0625, 3.0625, 3.0625, 6.1250],
                    [6.1250, 6.1250, 9.1875, 9.1875, 9.1875, 12.2500],
                    [12.2500, 12.2500, 15.3125, 15.3125, 15.3125, 18.3750],
                    [18.3750, 18.3750, 21.5000, 21.5000, 21.5000, 21.5000],
                ],
                dtype=torch.bfloat16,
            ),
            0.85,
        ),
        # token is not supported
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="channel",
            ),
            torch.tensor([[0], [6], [12], [18]]),
            torch.tensor([[5], [11], [17], [23]]),
            torch.tensor(
                [
                    [0.0000, 1.3359, 2.0000, 2.6719, 4.0000, 4.6875],
                    [5.8750, 7.3438, 7.3438, 8.8125, 10.2500, 10.2500],
                    [11.3125, 13.6250, 13.6250, 15.8750, 15.8750, 15.8750],
                    [18.3750, 18.3750, 21.5000, 21.5000, 21.5000, 21.5000],
                ],
                dtype=torch.bfloat16,
            ),
            0.45,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=3,
            ),
            torch.tensor([[0, 3], [6, 9], [12, 15], [18, 21]]),
            torch.tensor([[2, 5], [8, 11], [14, 17], [20, 23]]),
            torch.tensor(
                [
                    [0.0000, 1.0703, 1.8750, 2.6719, 4.0000, 4.6875],
                    [6.4375, 7.5000, 7.5000, 8.8125, 10.2500, 10.2500],
                    [11.1875, 13.0625, 13.0625, 15.8750, 15.8750, 15.8750],
                    [18.7500, 18.7500, 18.7500, 21.5000, 21.5000, 21.5000],
                ],
            ),
            0.45,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="float",  # tensor group requires FP4
                symmetric=True,
                strategy="tensor_group",  # requires float4
                group_size=3,
            ),
            torch.tensor([[0, 3], [6, 9], [12, 15], [18, 21]]),
            torch.tensor([[2, 5], [8, 11], [14, 17], [20, 23]]),
            torch.tensor(
                [
                    [0.0000, 1.0234, 2.0469, 3.2812, 3.2812, 4.9375],
                    [5.4688, 8.1875, 8.1875, 10.6875, 10.6875, 10.6875],
                    [9.8750, 14.7500, 14.7500, 16.3750, 16.3750, 16.3750],
                    [19.7500, 19.7500, 19.7500, 23.0000, 23.0000, 23.0000],
                ],
            ),
            1.1,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="block",
                block_structure=[2, 3],
            ),
            torch.tensor([[0, 3], [12, 15]]),
            torch.tensor([[8, 11], [20, 23]]),
            torch.tensor(
                [
                    [0.0000, 1.0703, 2.1406, 2.9375, 4.4062, 4.4062],
                    [6.4375, 7.5000, 7.5000, 8.8125, 10.2500, 10.2500],
                    [10.6875, 13.3750, 13.3750, 15.3125, 15.3125, 18.3750],
                    [18.7500, 18.7500, 18.7500, 21.5000, 21.5000, 21.5000],
                ],
            ),
            0.5,
        ),
    ],
)
def test_static_weight_quantization(
    args, exp_min_val, exp_max_val, exp_quant, exp_loss
):
    """
    weight = tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    """
    # set up weight
    input_size, output_size = 6, 4
    linear = torch.nn.Linear(input_size, output_size, bias=False)
    linear.weight.data = torch.arange(
        input_size * output_size, dtype=torch.bfloat16
    ).reshape(output_size, input_size)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme
    linear.weight_observer = MockMinMaxObserver("weight", args, linear)

    # calibrate_global_scale
    if hasattr(linear, "weight_global_scale"):
        global_scale = linear.weight_observer.get_global_scale(linear.weight)
        linear.weight_global_scale.data = global_scale

    # calibrate quantization parameters
    scale, zero_point = linear.weight_observer(linear.weight)
    linear.weight_scale.data = scale
    linear.weight_zero_point.data = zero_point
    assert torch.equal(linear.weight_observer.min_vals, exp_min_val)
    assert torch.equal(linear.weight_observer.max_vals, exp_max_val)

    # forward pass
    input = torch.eye(input_size, dtype=torch.bfloat16)
    output = linear(input)

    assert torch.allclose(output.T, exp_quant.to(output.dtype))
    assert torch.nn.functional.mse_loss(output.T, linear.weight) <= exp_loss


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_quant,exp_loss",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",
            ),
            torch.tensor([0.0]),
            torch.tensor([11.0]),
            torch.tensor(
                [
                    [
                        [0.0000, 1.4688, 1.4688, 2.9375, 4.4062, 4.4062],
                        [5.8750, 7.3438, 7.3438, 8.8125, 10.2500, 10.2500],
                    ]
                ]
            ),
            0.2,
        ),
        # static token is not supported
        # channel is not supported
        # group is not supported
        (
            QuantizationArgs(
                num_bits=4,
                type="float",  # must be fp4
                symmetric=True,
                strategy="tensor_group",
                dynamic="local",
                group_size=3,
            ),
            None,
            None,
            torch.tensor(
                [
                    [
                        [0.0000, 0.9844, 1.9688, 3.4062, 3.4062, 5.1250],
                        [5.2500, 7.8750, 7.8750, 7.3438, 11.0000, 11.0000],
                    ]
                ]
            ),
            0.5,
        ),
        # block is not supported
        # head is not supported
    ],
)
def test_static_activation_quantization(
    args, exp_min_val, exp_max_val, exp_quant, exp_loss
):
    """
    input = tensor([[ 0,  1,  2,  3,  4,  5]
                    [ 6,  7,  8,  9, 10, 11]])
    """
    # set up activation (and identity weight)
    batch_size, seq_len, input_size = 1, 2, 6
    input = torch.arange(
        (batch_size * seq_len * input_size), dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, input_size))
    linear = torch.nn.Linear(input_size, input_size, bias=False)
    linear.weight.data = torch.eye(input_size, dtype=torch.bfloat16)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme
    linear.input_observer = MockMinMaxObserver("input", args, linear)

    # calibrate quantization parameters
    def calibrate_input_hook(_, args):
        if hasattr(linear, "input_global_scale"):
            global_scale = linear.input_observer.get_global_scale(args[0])
            linear.input_global_scale.data = global_scale

        if linear.quantization_scheme.input_activations.dynamic is False:
            scale, zero_point = linear.input_observer(args[0])
            linear.input_scale.data = scale
            linear.input_zero_point.data = zero_point

    linear.register_forward_pre_hook(calibrate_input_hook)

    # calibration forward pass
    output = linear(input)

    # check calibration
    if exp_min_val is not None:
        assert torch.equal(linear.input_observer.min_vals, exp_min_val)
    if exp_max_val is not None:
        assert torch.equal(linear.input_observer.max_vals, exp_max_val)

    # check forward pass
    assert torch.allclose(output, exp_quant.to(output.dtype))
    assert torch.nn.functional.mse_loss(output, input) <= exp_loss


class MockAttention(torch.nn.Module):
    pass


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_quant,exp_loss",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",
            ),
            torch.tensor([0.0]),
            torch.tensor([11.0]),
            torch.tensor(
                [
                    [
                        [[0.0000, 1.4688, 1.4688], [2.9375, 4.4062, 4.4062]],
                        [[5.8750, 7.3438, 7.3438], [8.8125, 10.2500, 10.2500]],
                    ]
                ]
            ),
            0.19,
        ),
        # static token is not supported
        # channel is not supported
        # group is not supported
        # tensor group is not supported
        # block is not supported
    ],
)
def test_static_attention_quantization(
    args, exp_min_val, exp_max_val, exp_quant, exp_loss
):
    """
    input = tensor([[[[ 0.,  1.,  2.],
                      [ 3.,  4.,  5.]],

                      [[ 6.,  7.,  8.],
                      [ 9., 10., 11.]]]])
    """
    # set up activation (and identity weight)
    batch_size, seq_len, num_heads, head_dim = 1, 2, 2, 3
    input = torch.arange(
        (batch_size * seq_len * num_heads * head_dim), dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, num_heads, head_dim))
    attention = MockAttention()

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_qparams(
        attention, "k", args, (num_heads, head_dim), observed_dtype=torch.bfloat16
    )
    attention.quantization_scheme = scheme
    attention.quantization_status = QuantizationStatus.INITIALIZED
    attention.k_observer = MockMinMaxObserver("k", args, attention)

    # calibrate quantization parameters
    if scheme.input_activations.dynamic is False:
        scale, zero_point = attention.k_observer(input)
        attention.k_scale.data = scale
        attention.k_zero_point.data = zero_point

    # calibration forward pass
    output = forward_quantize(attention, input, "k", scheme.input_activations)

    # check calibration
    if exp_min_val is not None:
        assert torch.equal(attention.k_observer.min_vals, exp_min_val)
    if exp_max_val is not None:
        assert torch.equal(attention.k_observer.max_vals, exp_max_val)

    # check forward pass
    assert torch.allclose(output, exp_quant.to(output.dtype))
    assert torch.nn.functional.mse_loss(output, input) <= exp_loss
