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
from compressed_tensors.transform import TransformArgs


class TransformableModel(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.fcs = torch.nn.ModuleList(
            [
                torch.nn.Linear(sizes[index], sizes[index + 1], bias=False)
                for index in range(0, len(sizes) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


@pytest.fixture(scope="function")
def model_apply():
    model = TransformableModel(2, 4, 8, 16, 32, 64)
    apply = [
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

    return model, apply
