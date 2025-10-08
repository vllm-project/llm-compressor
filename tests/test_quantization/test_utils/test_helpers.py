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
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import (
    calculate_qparams,
    compute_dynamic_scales_and_zp,
    generate_gparam,
)


@pytest.mark.parametrize(
    "keepdims,strategy,exp_shape",
    [
        (
            False,
            "tensor",
            torch.Size(
                [
                    1,
                ]
            ),
        ),
        (True, "channel", torch.Size([1, 1])),
        (True, "group", torch.Size([1, 1])),
        (
            False,
            "block",
            torch.Size(
                [
                    1,
                ]
            ),
        ),
        (True, "token", torch.Size([1, 1])),
    ],
)
def test_calculate_qparams(keepdims, strategy, exp_shape):
    value = torch.empty(5, 6)
    min_val = torch.amin(value, dim=tuple(), keepdims=keepdims)
    max_val = torch.amax(value, dim=tuple(), keepdims=keepdims)

    if strategy == QuantizationStrategy.GROUP:
        args = QuantizationArgs(strategy=strategy, group_size=2)
    elif strategy == QuantizationStrategy.BLOCK:
        args = QuantizationArgs(strategy=strategy, block_structure=[1, 3])
    else:
        args = QuantizationArgs(
            strategy=strategy,
            group_size=(2 if strategy == "group" else None),
            block_structure=([1, 3] if strategy == "block" else None),
        )
        scale, zp = calculate_qparams(min_val, max_val, args)
        assert scale.shape == exp_shape
        assert zp.shape == exp_shape


def test_fused_global_scales():
    layer = torch.nn.Linear(7, 8)
    max_tensor_value = torch.abs(layer.weight.data).max()
    # use defaults
    min_val, max_val = torch.aminmax(layer.weight)
    global_scale = generate_gparam(min_val.data, max_val.data)
    # max value should be = (448 * 6) / global_scale
    assert max_tensor_value.item() == pytest.approx(
        FP4_E2M1_DATA.max * FP8_E4M3_DATA.max / global_scale, abs=0.001
    )


@pytest.mark.parametrize(
    "shape,group_size,exp_shape",
    [
        # Only batch size =1 is supported for dynamic GROUP quantization
        ((1, 4, 8), 4, torch.Size([1, 4, 2])),
    ],
)
def test_compute_dynamic_scales_and_zp_group(shape, group_size, exp_shape):
    """
    Dynamic group quantization should reduce activations in groups, producing
    scales and zero points of shape [batch, num_groups].
    """
    value = torch.randn(*shape)
    args = QuantizationArgs(
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        dynamic=True,
    )
    scale, zp = compute_dynamic_scales_and_zp(value, args, module=torch.nn.Module())
    assert scale.shape == exp_shape
    assert zp.shape == exp_shape
