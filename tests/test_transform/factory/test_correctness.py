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
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
)
from compressed_tensors.utils import offloaded_dispatch
from tests.testing_utils import requires_accelerate, requires_gpu


def scheme_kwargs():
    all_types = TransformFactory.registered_names()
    base = [{"type": type} for type in all_types]
    randomized = [{"type": type, "randomize": True} for type in all_types]
    return base + randomized


@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_correctness_linear(scheme_kwargs):
    size = (4, 8)
    module = torch.nn.Linear(*size, bias=True)
    scheme = TransformScheme(**scheme_kwargs)
    factory = TransformFactory.from_scheme(scheme, name="")

    input_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="input", inverse=True)
    )
    w_in_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_input")
    )
    w_out_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_output")
    )
    output_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="output", inverse=True)
    )

    input = torch.rand((17, size[0]))
    true_output = input @ module.weight.T
    input_transformed = input_tfm(input)
    weight_transformed = w_out_tfm(w_in_tfm(module.weight))
    output = output_tfm(input_transformed @ weight_transformed.T)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_correctness_model(scheme_kwargs, model_apply, offload=False):
    # load model
    model = model_apply[0]
    if offload:
        model = offloaded_dispatch(model, torch.device("cuda"))

    # get output
    input = torch.rand((17, model.fcs[0].in_features))
    if offload:
        input = input.to(torch.device("cuda"))
    true_output = model(input)

    # apply transforms
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                **scheme_kwargs,
                apply=model_apply[1],
            )
        }
    )
    for name, scheme in config.config_groups.items():
        factory = TransformFactory.from_scheme(scheme, name=name)
        factory.apply_to_model(model)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("scheme_kwargs", scheme_kwargs())
def test_correctness_model_offload(scheme_kwargs, model_apply):
    test_correctness_model(scheme_kwargs, model_apply, offload=True)
