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
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Linear


NUM_BITS = 8


@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            None,
        ),
        (
            None,
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
    ],
)
def test_initialize_module_for_quantization(
    create_quantization_scheme, weights, input_activations
):
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=weights,
        input_activations=input_activations,
    )
    layer = Linear(4, 4)

    assert not hasattr(layer, "quantization_scheme")
    assert not hasattr(layer, "quantization_status")

    # add attributes, zero_points and scale
    initialize_module_for_quantization(layer, quantization_scheme)

    registered_params = {"weight", "bias"}
    if weights is not None:
        registered_params.add("weight_scale")
        registered_params.add("weight_zero_point")

    if input_activations is not None:
        registered_params.add("input_scale")
        registered_params.add("input_zero_point")

    for key in layer.state_dict().keys():
        assert key in registered_params
        registered_params.remove(key)

    assert len(registered_params) == 0

    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")

    assert layer.quantization_status == QuantizationStatus.INITIALIZED
