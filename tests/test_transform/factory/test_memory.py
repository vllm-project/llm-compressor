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
    TransformFactory,
    TransformScheme,
)
from compressed_tensors.utils import align_modules, offloaded_dispatch
from tests.test_transform.conftest import TransformableModel
from tests.testing_utils import requires_accelerate, requires_gpu


def scheme_kwargs():
    all_types = TransformFactory.registered_names()
    base = [{"type": type} for type in all_types]
    randomized = [{"type": type, "randomize": True} for type in all_types]
    return base + randomized


@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_memory_sharing(scheme_kwargs, offload=False):
    # load model (maybe with offloading)
    model = TransformableModel(2, 2, 4, 4, 8, 8)
    if offload:
        offloaded_dispatch(model, torch.device("cuda"))

    # add transforms to model
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                **scheme_kwargs,
                apply=[
                    TransformArgs(targets="Linear", location="input"),
                    TransformArgs(targets="Linear", location="output"),
                ],
            )
        }
    )
    for name, scheme in config.config_groups.items():
        factory = TransformFactory.from_scheme(scheme, name=name)
        factory.apply_to_model(model)

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
@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_memory_sharing_offload(scheme_kwargs):
    test_memory_sharing(scheme_kwargs, offload=True)


@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_memory_sharing_training(scheme_kwargs):
    scheme_kwargs["requires_grad"] = True
    test_memory_sharing(scheme_kwargs, offload=False)
