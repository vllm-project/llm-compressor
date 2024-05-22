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


@pytest.mark.parametrize(
    "symmetric,expected_scale,expected_zero_point",
    [
        (True, 0.0078, 0),
        (False, 0.0039, -128),
    ],
)
def test_min_max_observer(symmetric, expected_scale, expected_zero_point):
    tensor = torch.tensor([1, 1, 1, 1, 1])
    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=symmetric)

    observer = weights.get_observer()
    scale, zero_point = observer(tensor)

    assert round(scale.item(), 4) == expected_scale
    assert round(zero_point.item(), 4) == expected_zero_point


def test_min_max_observer_symmetric_scale_range():
    tensor = torch.rand(4, 4)
    tensor *= 127

    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True)

    observer = weights.get_observer()
    scale, zero_point = observer(tensor)

    # if symmetric, max symmetric_range = abs(-128) / 255
    assert round(scale.item(), 4) <= 1.0039
    assert round(zero_point.item(), 4) == 0


def test_min_max_observer_value_update():
    inp = torch.tensor([1, 1, 1, 1, 1])
    inp_update_max = torch.tensor([127, 1, 1, 1, 1])
    inp_update_min = torch.tensor([-128, 1, 1, 1, 1])

    delta = 1e-6

    # udpate the min, max twice total
    tensors = [
        inp,
        inp,
        inp_update_max,  # update max
        inp,
        inp_update_min,  # update min
    ]

    tensor = inp
    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True)

    observer = weights.get_observer()
    curr_max = 1
    curr_min = 1
    for i, tensor in enumerate(tensors):
        observer(tensor)
        curr_max = max(observer.max_val.get("default"), curr_max)
        curr_min = min(observer.min_val.get("default"), curr_max)

        if i < 2:
            assert curr_max == 1
            assert curr_min == 1
        elif i < 4:
            assert abs(curr_max - 2.2600) < delta
            assert curr_min == 1
        else:
            assert abs(curr_max - 2.2600) < delta
            assert abs(curr_min - (-0.2900)) < delta
