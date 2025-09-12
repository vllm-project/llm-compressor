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

from collections import OrderedDict

import pytest
import torch
from compressed_tensors.config.format import (
    infer_and_set_per_module_quantization_format,
)
from compressed_tensors.quantization import preset_name_to_scheme


@pytest.mark.parametrize(
    "preset,sparsity_structure,expected_format",
    [
        ["W8A8", "unstructured", "int-quantized"],
        ["W8A16", "unstructured", "pack-quantized"],
        ["W8A16", "2:4", "marlin-24"],
        ["W4A16", "unstructured", "pack-quantized"],
        ["W4A16", "2:4", "marlin-24"],
        ["FP8", "unstructured", "float-quantized"],
    ],
)
def test_infer_quant_format(preset, sparsity_structure, expected_format):
    quant_scheme = preset_name_to_scheme(preset, targets=["Linear"])

    dummy_model = torch.nn.Sequential(
        OrderedDict(
            [
                ("fc1", torch.nn.Linear(8, 16, bias=True)),
                ("fc2", torch.nn.Linear(16, 32, bias=True)),
                (
                    "block1",
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("fc1", torch.nn.Linear(32, 16, bias=True)),
                                ("fc2", torch.nn.Linear(16, 8, bias=True)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    for _, module in dummy_model.named_modules():
        module.quantization_scheme = quant_scheme

    inferred_format = infer_and_set_per_module_quantization_format(
        dummy_model, sparsity_structure=sparsity_structure
    )
    assert inferred_format[0] == expected_format
