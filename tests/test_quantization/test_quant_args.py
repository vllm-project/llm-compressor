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
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import ValidationError


def test_defaults():
    default = QuantizationArgs()

    assert default.num_bits == 8
    assert default.type == QuantizationType.INT
    assert default.symmetric
    assert default.strategy == QuantizationStrategy.TENSOR
    assert default.group_size is None
    assert default.block_structure is None


def test_group():
    kwargs = {"strategy": "group", "group_size": 128}

    group = QuantizationArgs(**kwargs)
    assert group.strategy == QuantizationStrategy.GROUP
    assert group.group_size == kwargs["group_size"]

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP, group_size=-1)

    args = QuantizationArgs(group_size=128, strategy="group")
    assert args.group_size == 128
    assert args.strategy == "group"

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP)

    with pytest.raises(ValueError):
        QuantizationArgs(strategy="tensor", group_size=128)


def test_block():
    kwargs = {"strategy": "block", "block_structure": "2x4"}

    block = QuantizationArgs(**kwargs)
    assert block.strategy == QuantizationStrategy.BLOCK
    assert block.block_structure == kwargs["block_structure"]


def test_infer_strategy():
    args = QuantizationArgs(group_size=128)
    assert args.strategy == QuantizationStrategy.GROUP

    args = QuantizationArgs(group_size=-1)
    assert args.strategy == QuantizationStrategy.CHANNEL


def test_enums():
    assert QuantizationArgs(
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        actorder=ActivationOrdering.WEIGHT,
        group_size=1,
    ) == QuantizationArgs(type="InT", strategy="GROUP", actorder="weight", group_size=1)


def test_actorder():
    # test group inference with actorder
    args = QuantizationArgs(group_size=128, actorder=ActivationOrdering.GROUP)
    assert args.strategy == QuantizationStrategy.GROUP

    # test invalid pairings
    with pytest.raises(ValueError):
        QuantizationArgs(group_size=None, actorder="weight")
    with pytest.raises(ValueError):
        QuantizationArgs(group_size=-1, actorder="weight")
    with pytest.raises(ValueError):
        QuantizationArgs(strategy="tensor", actorder="weight")

    # test boolean and none defaulting
    assert (
        QuantizationArgs(group_size=1, actorder=True).actorder
        == ActivationOrdering.GROUP
    )
    assert QuantizationArgs(group_size=1, actorder=False).actorder is None
    assert QuantizationArgs(group_size=1, actorder=None).actorder is None


def test_invalid():
    with pytest.raises(ValidationError):
        QuantizationArgs(type="invalid")
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy="invalid")
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP)
