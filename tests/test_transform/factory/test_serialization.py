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

import pytest
import torch
from compressed_tensors.transform import (
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import offloaded_dispatch
from safetensors import safe_open
from tests.testing_utils import requires_accelerate, requires_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    model_path = os.path.join(tmp_path, "test_model_path")
    model.save_pretrained(model_path)

    # check that saved values match model values
    # note that shared weights are only serialized once
    safetensors_path = os.path.join(model_path, "model.safetensors")
    with safe_open(safetensors_path, framework="pt", device="cpu") as file:
        saved_keys = set(file.keys())
        assert {
            "fcs.0.weight",
            "fcs.1.weight",
            "fcs.2.weight",
            "fcs.3.weight",
            "fcs.4.weight",
        } <= saved_keys
        for key in saved_keys:
            param = model.get_parameter(key)
            saved_param = file.get_tensor(key)

            if param.device.type != "meta":  # skip testing values in offload case
                assert torch.equal(param, saved_param)


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
def test_serialization_offload(type, randomize, model_apply, tmp_path):
    test_serialization(type, randomize, model_apply, tmp_path, offload=True)


@pytest.mark.skip("Requires transformers#40673")
@requires_gpu
@pytest.mark.parametrize(
    "model_stub,exp_perplexity",
    [
        ("nm-testing/Llama-3.2-1B-Instruct-spinquantR1R2R4-w4a16", 10.0),
        ("nm-testing/Llama-3.2-1B-Instruct-quip-w4a16", 10.0),
    ],
)
def test_load_perplexity(model_stub, exp_perplexity):
    model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    prompt = "The capital of France is Paris, the capital of Germany is Berlin"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    labels = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    perplexity = torch.exp(outputs.loss)
    assert perplexity <= exp_perplexity
