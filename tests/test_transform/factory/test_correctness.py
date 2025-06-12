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
    TransformFactory,
    TransformScheme,
)
from compressed_tensors.utils import offloaded_dispatch
from tests.testing_utils import requires_accelerate, requires_gpu


class TransformableModel(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.fcs = torch.nn.ModuleList([])
        self.fcs.append(torch.nn.Linear(sizes[0], sizes[1], bias=False))
        for index in range(1, len(sizes) - 1):
            self.fcs.append(torch.nn.Linear(sizes[index], sizes[index + 1], bias=False))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


@pytest.mark.parametrize(
    "scheme",
    [TransformScheme(type=name) for name in TransformFactory.registered_names()],
)
def test_correctness_linear(scheme):
    size = (4, 8)
    module = torch.nn.Linear(*size, bias=True)
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


@pytest.mark.parametrize(
    "scheme",
    [TransformScheme(type=name) for name in TransformFactory.registered_names()],
)
def test_correctness_model(scheme, offload=False):
    # load model
    model = TransformableModel(2, 4, 8, 16, 32, 64)
    if offload:
        model = offloaded_dispatch(model, torch.device("cuda"))

    # create factory
    scheme.apply = [
        # weight output -> input
        TransformArgs(targets="fcs.0", location="weight_output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        # output -> weight input
        TransformArgs(targets="fcs.1", location="output"),
        TransformArgs(targets="fcs.2", location="weight_input", inverse=True),
        # output -> input
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.3", location="input", inverse=True),
        # weight output -> weight input
        TransformArgs(targets="fcs.3", location="weight_output"),
        TransformArgs(targets="fcs.4", location="weight_input", inverse=True),
    ]
    factory = TransformFactory.from_scheme(scheme, name="")

    # create inputs
    input = torch.rand((17, model.fcs[0].in_features))
    if offload:
        input = input.to(torch.device("cuda"))

    # compare outputs
    true_output = model(input)
    factory.apply_to_model(model)
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize(
    "scheme",
    [TransformScheme(type=name) for name in TransformFactory.registered_names()],
)
def test_correctness_model_offload(scheme):
    test_correctness_model(scheme, offload=True)
