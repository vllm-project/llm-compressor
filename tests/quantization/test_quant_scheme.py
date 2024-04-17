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
from pydantic import ValidationError
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme


def test_basic_scheme():
    targets = ["model.layer.0", "model.layer.3"]
    weights = QuantizationArgs()

    scheme = QuantizationScheme(targets=targets, weights=weights)
    assert scheme.targets == targets
    assert scheme.weights == weights
    assert scheme.input_activations is None
    assert scheme.output_activations is None


def test_full_scheme():
    targets = ["Linear"]
    weights = QuantizationArgs()
    input_activations = QuantizationArgs(num_bits=4)
    output_activations = QuantizationArgs(num_bits=8, type="float", symmetric=False)

    scheme = QuantizationScheme(
        targets=targets,
        weights=weights,
        input_activations=input_activations,
        output_activations=output_activations,
    )
    assert scheme.targets == targets
    assert scheme.weights == weights
    assert scheme.input_activations == input_activations
    assert scheme.output_activations == output_activations


def test_needs_targets():
    with pytest.raises(ValidationError):
        _ = QuantizationScheme()
