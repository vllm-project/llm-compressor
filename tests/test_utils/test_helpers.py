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

import os
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors import (
    ParameterizedDefaultDict,
    load_compressed,
    patch_attr,
    save_compressed,
    save_compressed_model,
)
from compressed_tensors.config import BitmaskConfig
from safetensors.torch import save_model
from transformers import AutoModelForCausalLM


@pytest.fixture
def tensors():
    tensors = {"tensor_1": torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
    return tensors


@pytest.fixture
def llama_model():
    return AutoModelForCausalLM.from_pretrained(
        "RedHatAI/llama2.c-stories110M-pruned50",
        torch_dtype="auto",
    )


def test_save_compressed_sparse_bitmask(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="sparse-bitmask",
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_dense(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="dense",
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_no_compression(tmp_path, tensors):
    save_compressed(
        tensors,
        save_path=tmp_path / "model.safetensors",
    )
    assert (tmp_path / "model.safetensors").exists()


def test_save_compressed_error(tmp_path):
    with pytest.raises(Exception):
        save_compressed({}, "")

    with pytest.raises(Exception):
        save_compressed(None, "")

    with pytest.raises(Exception):
        save_compressed(
            tensors,
            compression_format="this_is_not_a_valid_format",
            save_path=tmp_path / "model.safetensors",
        )


def test_load_compressed_sparse_bitmask(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="sparse-bitmask",
        save_path=tmp_path / "model.safetensors",
    )
    compression_config = BitmaskConfig(
        format="sparse-bitmask",
    )
    loaded_tensors = dict(
        load_compressed(tmp_path / "model.safetensors", compression_config)
    )
    for key in tensors:
        assert torch.allclose(tensors[key], loaded_tensors[key])


def test_load_compressed_dense(tmp_path, tensors):
    save_compressed(
        tensors,
        compression_format="dense",
        save_path=tmp_path / "model.safetensors",
    )
    save_compressed(
        tensors,
        save_path=tmp_path / "model_.safetensors",
    )

    loaded_tensors = dict(load_compressed(tmp_path / "model.safetensors"))
    loaded_tensors_ = dict(load_compressed(tmp_path / "model_.safetensors"))
    # loaded_tensors should be equal to loaded_tensors_
    for key in tensors:
        assert torch.allclose(loaded_tensors[key], loaded_tensors_[key])


def test_load_compressed_sharded(tmp_path, llama_model):
    sharded_model_path = tmp_path / "sharded_model"
    llama_model.save_pretrained(sharded_model_path, max_shard_size="2MB")
    # make sure that model is sharded on disk
    assert len(os.listdir(sharded_model_path)) > 1
    loaded_state_dict = dict(load_compressed(sharded_model_path))
    for key, value in llama_model.state_dict().items():
        if key == "lm_head.weight":
            # lm_head doesn't have separate weights.
            # It shares its weight tensor with the token embedding layer.
            continue
        assert torch.allclose(value, loaded_state_dict[key])


def test_save_compressed_model(tmp_path, llama_model):
    path_to_uncompressed = tmp_path / "model_uncompressed.safetensors"
    path_to_compressed = tmp_path / "model_compressed.safetensors"

    # save uncompressed model
    save_model(llama_model, path_to_uncompressed)
    size_uncompressed_kb = path_to_uncompressed.stat().st_size / 1024

    # save compressed model
    save_compressed_model(
        llama_model, path_to_compressed, compression_format="sparse-bitmask"
    )
    size_compressed_kb = path_to_compressed.stat().st_size / 1024

    # compare that the are the same after loading
    state_dict_1 = dict(load_compressed(path_to_uncompressed))
    state_dict_2 = dict(
        load_compressed(path_to_compressed, BitmaskConfig(format="sparse-bitmask"))
    )
    assert all(
        torch.allclose(state_dict_1[key], state_dict_2[key]) for key in state_dict_1
    )
    # make sure that compressed model is smaller
    # than uncompressed by roughly 1.14 (value established empirically)
    assert pytest.approx(size_uncompressed_kb / size_compressed_kb, 0.01) == 1.14


def test_patch_attr():
    # patch, original value
    obj = SimpleNamespace()
    obj.attribute = "original"
    with patch_attr(obj, "attribute", "patched"):
        assert obj.attribute == "patched"
        obj.attribute = "modified"
    assert obj.attribute == "original"

    # patch, no original attribute
    obj = SimpleNamespace()
    with patch_attr(obj, "attribute", "patched"):
        assert obj.attribute == "patched"
        obj.attribute = "modified"
    assert not hasattr(obj, "attribute")


def test_parameterized_default_dict():
    def add_one(value):
        return value + 1

    add_dict = ParameterizedDefaultDict(add_one)
    assert add_dict[0] == 1
    assert add_dict[1] == 2

    def sum_vals(a, b):
        return a + b

    sum_dict = ParameterizedDefaultDict(sum_vals)
    assert sum_dict[0, 1] == 1
    assert sum_dict[5, 7] == 12
