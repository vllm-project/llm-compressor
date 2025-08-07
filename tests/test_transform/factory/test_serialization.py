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
from compressed_tensors.transform import (
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import offloaded_dispatch
from tests.testing_utils import requires_accelerate, requires_gpu


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
def test_serialization(type, randomize, model_apply, tmp_path, offload=False):
    # get model, maybe offload
    model, apply = model_apply
    if offload:
        offloaded_dispatch(model, torch.device("cuda"))

    # apply transforms to model
    config = TransformConfig(
        config_groups={"": TransformScheme(type=type, randomize=randomize, apply=apply)}
    )
    apply_transform_config(model, config)

    # save model
    model.save_pretrained(tmp_path)

    # TODO: reload model


@pytest.mark.skip(reason="Requires changes in upstream transformers")
# https://github.com/huggingface/transformers/pull/39280
# https://github.com/huggingface/transformers/pull/39263
@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
def test_serialization_offload(type, randomize, model_apply, tmp_path):
    test_serialization(type, randomize, model_apply, tmp_path, offload=True)
