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
from compressed_tensors.linear.compressed_linear import CompressedLinear
from transformers import AutoModelForCausalLM, AutoTokenizer


def models_with_linear_quantized():
    return [
        # weights packed
        "nm-testing/llama2.c-stories110M-gsm8k-recipe_w4a16_actorder_weight-compressed",
        # weights not packed
        "nm-testing/llama2.c-stories110M-gsm8k-fp8_dynamic-compressed",
    ]


@pytest.mark.parametrize("model_stub", models_with_linear_quantized())
def test_model_forward_pass(model_stub):
    """
    Test that AutoModelForCausalLM can process tokenized inputs and generate output.
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.float16, device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    # Define sample input
    sample_inputs = [
        "I love quantization because",
        "What is the capital of France?",
        "def fibonacci(n):",
    ]

    # Move inputs to the correct device
    device = next(model.parameters()).device
    inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True).to(device)

    # Run model inference (forward pass)
    outputs = model.generate(**inputs, max_length=50)

    # Ensure output is not empty
    assert outputs is not None, "Model forward pass failed, no output generated."


@pytest.mark.parametrize("model_stub", models_with_linear_quantized())
def test_compressed_linear_from_linear_usage(monkeypatch, model_stub):
    """
    Test that CompressedLinear.from_linear is used for creating
    CompressedLinear instances.
    """
    call_count = 0

    original_from_linear = CompressedLinear.from_linear

    def fake_from_linear(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_from_linear(*args, **kwargs)

    # Replace the original from_linear with our fake to count its invocations
    monkeypatch.setattr(CompressedLinear, "from_linear", fake_from_linear)

    # Load model to trigger the creation of CompressedLinear instances
    model = AutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype="auto", device_map="auto"
    )

    # Known quantized layers that should be
    # instances of CompressedLinear
    # (This is not an exhaustive list)
    quantized_layers = {"q_proj", "k_proj", "v_proj"}

    # Check that the expected layers are instances of CompressedLinear
    for layer_name, module in model.named_modules():
        if any(layer in layer_name for layer in quantized_layers):
            assert isinstance(
                module, CompressedLinear
            ), f"{layer_name} should be an instance of CompressedLinear"
            f"but got {type(module).__name__}"

    assert call_count > 0, "`CompressedLinear.from_linear` was not used during the "
    "creation of CompressedLinear instances."
