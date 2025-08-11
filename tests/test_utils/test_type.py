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
from compressed_tensors.utils.type import TorchDtype
from pydantic import BaseModel, Field
from pydantic_core._pydantic_core import ValidationError


class DummyModel(BaseModel):
    dtype: TorchDtype = Field(default=torch.float32)


@pytest.mark.unit
def test_default_value():
    model = DummyModel()
    assert model.dtype == torch.float32


@pytest.mark.unit
def test_value_override():
    model = DummyModel()
    model.dtype = torch.float16
    assert model.dtype == torch.float16


@pytest.mark.unit
def test_validation():
    DummyModel(dtype=torch.float16)
    DummyModel(dtype="torch.float16")
    DummyModel(dtype="float16")

    with pytest.raises(ValidationError):
        model = DummyModel(dtype="notatype")


@pytest.mark.unit
def test_serialization():
    model = DummyModel()
    assert model.model_dump()["dtype"] == "torch.float32"
    assert DummyModel.model_validate(model.model_dump()) == model

    model = DummyModel(dtype=torch.float16)
    assert model.model_dump()["dtype"] == "torch.float16"
    assert DummyModel.model_validate(model.model_dump()) == model

    model = DummyModel()
    model.dtype = torch.float16
    assert model.model_dump()["dtype"] == "torch.float16"
    assert DummyModel.model_validate(model.model_dump()) == model


@pytest.mark.unit
def test_deserialization():
    dummy_dict = {"dtype": "torch.float16"}
    assert DummyModel.model_validate(dummy_dict).dtype == torch.float16

    dummy_dict = {"dtype": "float16"}
    assert DummyModel.model_validate(dummy_dict).dtype == torch.float16

    with pytest.raises(ValueError):
        dummy_dict = {"dtype": "notatype"}
        DummyModel.model_validate(dummy_dict)

    with pytest.raises(ValueError):
        dummy_dict = {"dtype": "torch.notatype"}
        DummyModel.model_validate(dummy_dict)
