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

from collections import Counter

import pytest
import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformBase,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import align_modules, offloaded_dispatch
from tests.test_transform.conftest import TransformableModel
from tests.testing_utils import requires_accelerate, requires_gpu


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_memory_sharing(type, randomize, requires_grad, offload=False):
    # load model (maybe with offloading)
    model = TransformableModel(2, 2, 4, 4, 8, 8)
    if offload:
        offloaded_dispatch(model, torch.device("cuda"))

    # add transforms to model
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomzied=randomize,
                requires_grad=requires_grad,
                apply=[
                    TransformArgs(targets="Linear", location="input"),
                    TransformArgs(targets="Linear", location="output"),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    # check that memory is shared when onloaded
    with align_modules(model.modules()):
        weights = [m.weight for m in model.modules() if isinstance(m, TransformBase)]
        weight_to_count = Counter(weights)
        size_to_weight = {weight.size(0): weight for weight in weight_to_count}

        assert len(weight_to_count) == len(size_to_weight) == 3
        assert weight_to_count[size_to_weight[2]] == 3
        assert weight_to_count[size_to_weight[4]] == 4
        assert weight_to_count[size_to_weight[8]] == 3

    # check that memory is shared in offloaded dict
    if offload:
        weights_map = dict(model.fcs[0]._hf_hook.weights_map.dataset)
        offloaded_weights = [
            value
            for name, value in weights_map.items()
            if name.endswith("_input.weight") or name.endswith("_output.weight")
        ]
        weight_to_count = Counter(offloaded_weights)
        size_to_weight = {weight.size(0): weight for weight in weight_to_count}

        assert len(weight_to_count) == len(size_to_weight) == 3
        assert weight_to_count[size_to_weight[2]] == 3
        assert weight_to_count[size_to_weight[4]] == 4
        assert weight_to_count[size_to_weight[8]] == 3


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
def test_memory_sharing_offload(
    type,
    randomize,
):
    test_memory_sharing(type, randomize, requires_grad=False, offload=True)
