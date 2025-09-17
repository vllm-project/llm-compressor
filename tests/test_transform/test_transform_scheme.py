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

from compressed_tensors.transform import TransformArgs, TransformScheme


def test_basic_scheme():
    targets = ["Linear"]
    location = "input"
    basic_args = TransformArgs(targets=targets, location=location)

    scheme = TransformScheme(
        type="hadamard",
        apply=[basic_args],
    )
    assert not scheme.randomize
    assert scheme.type == "hadamard"
    assert len(scheme.apply) == 1
    assert isinstance(scheme.apply[0], TransformArgs)


def test_multiple_groups_global():
    targets = ["Embedding"]
    location = "input"
    embedding_args = TransformArgs(targets=targets, location=location)

    targets = ["Linear"]
    location = "weight_input"
    linear_args = TransformArgs(targets=targets, location=location)

    # same transform applied to multiple groups
    scheme = TransformScheme(
        type="hadamard",
        apply=[embedding_args, linear_args],
        randomize=True,
    )

    assert scheme.randomize
    assert scheme.type == "hadamard"
    assert len(scheme.apply) == 2
    assert isinstance(scheme.apply[0], TransformArgs)
    assert isinstance(scheme.apply[1], TransformArgs)


def test_multiple_groups():
    apply = []
    location = "weight_output"

    for i in range(20):
        targets = [f"model.layers.{i}.attn.v_proj", f"model.layers.{i}.attn.o_proj"]
        args = TransformArgs(targets=targets, location=location)
        apply.append(args)

    # global is False, different hadamard transform applied to each group
    # same dimension/hidden dim
    scheme = TransformScheme(
        type="hadamard",
        apply=apply,
    )

    assert not scheme.randomize
    assert scheme.type == "hadamard"
    assert len(scheme.apply) == 20


def test_transform_scheme_block_size():
    """
    Ensure json with (deprecated) `head_dim` or `block_size`
    both load up correctly and save with `block_size` field
    """

    old_scheme = TransformScheme.model_validate_json(
        '{"type": "hadamard", "head_dim": 128}'
    )
    assert old_scheme.block_size == 128
    assert old_scheme.model_dump()["block_size"] == 128
    old_scheme = TransformScheme(type="hadamard", head_dim=64)
    assert old_scheme.block_size == 64
    assert old_scheme.model_dump()["block_size"] == 64

    new_scheme = TransformScheme.model_validate_json(
        '{"type": "hadamard", "block_size": 128}'
    )
    assert new_scheme.block_size == 128
    assert new_scheme.model_dump()["block_size"] == 128
    new_scheme = TransformScheme(type="hadamard", block_size=64)
    assert new_scheme.block_size == 64
    assert new_scheme.model_dump()["block_size"] == 64
