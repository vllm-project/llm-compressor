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
from compressed_tensors.quantization.lifecycle.forward import (
    maybe_calibrate_or_quantize,
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Linear


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
def test_maybe_calibrate_or_quantize(create_quantization_scheme, quantization_status):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )
    quantization_args = QuantizationArgs(num_bits=num_bits, symmetric=True)
    layer = Linear(4, 4)
    layer.weight.data *= 100
    layer.quantization_status = QuantizationStatus(quantization_status)

    initialize_module_for_quantization(layer, quantization_scheme)

    # only calibration updates the scale and zero-point
    if layer.quantization_status == QuantizationStatus.INITIALIZED:
        out = maybe_calibrate_or_quantize(
            layer, layer.weight.data, "input", quantization_args
        )
        assert torch.allclose(out, layer.weight.data)
    elif layer.quantization_status == QuantizationStatus.CALIBRATION:

        out = maybe_calibrate_or_quantize(
            layer, layer.weight.data, "input", quantization_args
        )
        assert torch.allclose(out, layer.weight.data, atol=0.2)

    elif layer.quantization_status == QuantizationStatus.FROZEN:
        # scale and zero points are empty -- cannot quantize
        with pytest.raises(Exception):
            out = maybe_calibrate_or_quantize(
                layer, layer.weight.data, "input", quantization_args
            )
