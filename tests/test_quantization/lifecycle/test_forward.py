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
from compressed_tensors.quantization.lifecycle.calibration import (
    set_module_for_calibration,
)
from compressed_tensors.quantization.lifecycle.forward import (
    calibrate_activations,
    dequantize,
    forward_quantize,
    quantize,
    wrap_module_forward_quantized,
    wrap_module_forward_quantized_attn,
)
from compressed_tensors.quantization.lifecycle.frozen import freeze_module_quantization
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Linear


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


def test_wrap_module_forward_quantized(create_quantization_scheme):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )
    layer = Linear(4, 4)

    func_forward = layer.forward.__func__

    # check that the forward call is overwritten
    wrap_module_forward_quantized(layer, quantization_scheme)

    assert not func_forward == layer.forward.__func__


@pytest.mark.parametrize(
    "quantization_status", ["initialized", "calibration", "frozen"]
)
def test_forward_quantize(create_quantization_scheme, quantization_status):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )
    quantization_args = QuantizationArgs(num_bits=num_bits, symmetric=True)
    layer = Linear(4, 4)
    layer.weight.data *= 100

    dummy_tensor = torch.randn(8, 4)  # (num_tokens, num_features)
    layer.quantization_status = QuantizationStatus(quantization_status)

    # only calibration updates the scale and zero-point
    if layer.quantization_status == QuantizationStatus.INITIALIZED:
        # Init zp and scales
        initialize_module_for_quantization(layer, quantization_scheme)
        # init weight observers; update weight scales/zp
        set_module_for_calibration(layer)
        # call quant/dequant on weights
        out = forward_quantize(layer, layer.weight, "weight", quantization_args)
        assert torch.allclose(out, layer.weight.data, atol=0.2)
    elif layer.quantization_status == QuantizationStatus.CALIBRATION:
        # init zp/scales
        initialize_module_for_quantization(layer, quantization_scheme)
        #  init weight observers; update weight scales/zp
        set_module_for_calibration(layer)
        #  init input observers, update input scales/zp
        calibrate_activations(
            module=layer,
            value=dummy_tensor,
            base_name="input",
            quantization_args=quantization_args,
        )
        # call quant/dequant on inputs
        out = forward_quantize(layer, dummy_tensor, "input", quantization_args)
        assert torch.allclose(out, dummy_tensor, atol=0.2)
        assert layer.input_observer._num_observed_tokens == dummy_tensor.shape[0]
    elif layer.quantization_status == QuantizationStatus.FROZEN:
        # init weight observers
        initialize_module_for_quantization(layer, quantization_scheme)
        # init weight observers; update weight scales/zp
        set_module_for_calibration(layer)
        # remove weight observers and any input observers
        freeze_module_quantization(layer)
        # call quant/dequant on weights
        out = forward_quantize(layer, layer.weight.data, "weight", quantization_args)
        assert torch.allclose(out, layer.weight.data, atol=0.2)


@pytest.mark.parametrize(
    "num_bits,type,strategy,group_size,scale,zero_point,g_idx",
    [
        (
            4,
            "int",
            QuantizationStrategy.TENSOR,
            None,
            torch.rand((1,)) * 0.01,
            torch.zeros((1,)),
            None,
        ),
        (
            4,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            None,
        ),
        (
            4,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            make_dummy_g_idx(1024, 128),
        ),
        (
            8,
            "float",
            QuantizationStrategy.TENSOR,
            None,
            torch.rand((1,)) * 0.01,
            torch.zeros((1,)),
            None,
        ),
        (
            8,
            "float",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            None,
        ),
        (
            8,
            "float",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            make_dummy_g_idx(1024, 128),
        ),
    ],
)
def test_quantize(num_bits, type, strategy, group_size, scale, zero_point, g_idx):
    args = QuantizationArgs(
        num_bits=num_bits, type=type, strategy=strategy, group_size=group_size
    )

    x = torch.rand((512, 1024))
    quantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=args.pytorch_dtype(),
        g_idx=g_idx,
    )


@pytest.mark.parametrize(
    "num_bits,type,strategy,group_size,scale,zero_point,g_idx",
    [
        (
            8,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            None,
        ),
        (
            8,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1)),
            make_dummy_g_idx(1024, 128),
        ),
    ],
)
def test_dequantize(num_bits, type, strategy, group_size, scale, zero_point, g_idx):
    args = QuantizationArgs(
        num_bits=num_bits, type=type, strategy=strategy, group_size=group_size
    )

    x_q = torch.rand((512, 1024)).to(dtype=args.pytorch_dtype())
    dequantize(
        x_q=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=None,
        g_idx=g_idx,
    )


def test_wrap_module_forward_quantized_attn(create_quantization_scheme):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["self_attn"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    mock_attn_layer = Linear(4, 4)

    attn_forward = mock_attn_layer.forward.__func__

    # check that the forward call is overwritten
    wrap_module_forward_quantized_attn(mock_attn_layer, quantization_scheme)

    assert not attn_forward == mock_attn_layer.forward.__func__
