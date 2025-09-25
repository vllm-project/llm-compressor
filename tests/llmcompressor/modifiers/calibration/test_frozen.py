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

from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Linear

from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
    initialize_observer,
)


def test_set_module_for_calibration():
    num_bits = 8
    quantization_scheme = QuantizationScheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    layer = Linear(4, 4)

    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = QuantizationStatus.CALIBRATION
    initialize_observer(layer, "weight")

    # should have both input and weight observer after initalizing
    assert hasattr(layer, "weight_observer")

    # observers should get deleted after freezing
    freeze_module_quantization(layer)
    assert not hasattr(layer, "input_observer")
    assert not hasattr(layer, "weight_observer")

    assert layer.quantization_status == QuantizationStatus.FROZEN
