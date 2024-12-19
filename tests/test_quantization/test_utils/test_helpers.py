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
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams


@pytest.mark.parametrize(
    "keepdims,strategy,exp_shape",
    [
        (
            False,
            QuantizationStrategy.TENSOR,
            torch.Size(
                [
                    1,
                ]
            ),
        ),
        (True, QuantizationStrategy.CHANNEL, torch.Size([1, 1])),
        (True, QuantizationStrategy.GROUP, torch.Size([1, 1])),
        (
            False,
            QuantizationStrategy.BLOCK,
            torch.Size(
                [
                    1,
                ]
            ),
        ),
        (True, QuantizationStrategy.TOKEN, torch.Size([1, 1])),
    ],
)
def test_calculate_qparams(keepdims, strategy, exp_shape):
    value = torch.randn(14, 5)
    min_val = torch.amin(value, dim=tuple(), keepdims=keepdims)
    max_val = torch.amax(value, dim=tuple(), keepdims=keepdims)

    if strategy == QuantizationStrategy.GROUP:
        args = QuantizationArgs(strategy=strategy, group_size=2)
    else:
        args = QuantizationArgs(strategy=strategy)
        scale, zp = calculate_qparams(min_val, max_val, args)
        assert scale.shape == exp_shape
        assert zp.shape == exp_shape
