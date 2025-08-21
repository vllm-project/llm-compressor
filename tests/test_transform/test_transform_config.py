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
from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme


@pytest.fixture(scope="module")
def scheme():
    targets = ["Embedding"]
    location = "input"
    basic_args = TransformArgs(targets=targets, location=location)

    return TransformScheme(
        type="hadamard",
        apply=[basic_args],
    )


@pytest.fixture(scope="module")
def config(scheme):
    return TransformConfig(
        config_groups={
            "transform_0": scheme,
        }
    )


def test_basic(config):
    assert isinstance(config.config_groups.get("transform_0"), TransformScheme)


def test_to_dict(config):
    config_dict = config.model_dump()
    assert "config_groups" in config_dict.keys()


def test_multiple_groups():
    location = "weight_input"

    targets_1 = ["model.layers.0.attn.v_proj"]
    linear_args_1 = TransformArgs(targets=targets_1, location=location)

    targets_2 = ["model.layers.0.attn.q_proj"]
    linear_args_2 = TransformArgs(targets=targets_2, location=location)

    scheme_1 = TransformScheme(
        type="hadamard",
        apply=[linear_args_1],
    )

    scheme_2 = TransformScheme(
        type="hadamard",
        apply=[linear_args_2],
    )
    _ = TransformConfig(
        config_groups={"transform_0": scheme_1, "transform_1": scheme_2}
    )


def test_reload(config):
    assert config == TransformConfig.model_validate(config.model_dump())
