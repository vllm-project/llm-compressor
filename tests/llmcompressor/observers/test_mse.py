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
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import MovingAverageMSEObserver, Observer


@pytest.mark.parametrize(
    "symmetric,expected_scale,expected_zero_point",
    [
        (True, 0.0078, 0),
        (False, 0.0039, -128),
    ],
)
def test_mse_observer(symmetric, expected_scale, expected_zero_point):
    tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=symmetric, observer="mse")

    observer = weights.observer
    observer = Observer.load_from_registry(observer, quantization_args=weights)
    scale, zero_point = observer(tensor)

    assert isinstance(observer, MovingAverageMSEObserver)
    assert round(scale.item(), 4) == expected_scale
    assert round(zero_point.item(), 4) == expected_zero_point


def test_mse_observer_symmetric_scale_range():
    tensor = torch.rand(4, 4)
    tensor *= 127

    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True, observer="mse")

    observer = weights.observer
    observer = Observer.load_from_registry(observer, quantization_args=weights)
    scale, zero_point = observer(tensor)

    # if symmetric, max symmetric_range = abs(-128) / 255
    assert round(scale.item(), 4) <= 1.0039
    assert round(zero_point.item(), 4) == 0
