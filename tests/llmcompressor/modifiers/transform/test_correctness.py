import os

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    untie_word_embeddings,
)
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
@pytest.mark.parametrize(
    "modifier,model_dtype,precision,transform_block_size,exp_mse",
    [
        (QuIPModifier, torch.bfloat16, torch.bfloat16, None, 5e-3),  # 0.0019
        (QuIPModifier, torch.bfloat16, torch.float32, 16, 5e-3),  # 0.0022
        (QuIPModifier, torch.float32, torch.float32, 32, 5e-10),  # 1.0e-10
        (QuIPModifier, torch.float32, torch.float64, 64, 5e-11),  # 2.7e-11
        (SpinQuantModifier, torch.bfloat16, torch.bfloat16, None, 5e-3),  # 0.0030
        (SpinQuantModifier, torch.bfloat16, torch.float32, 16, 5e-3),  # 0.0029
        (SpinQuantModifier, torch.float32, torch.float32, 32, 5e-4),  # 4e-4
        (SpinQuantModifier, torch.float32, torch.float64, 64, 5e-4),  # 4e-4
    ],
)
def test_apply_correctness(
    modifier, model_dtype, precision, transform_block_size, exp_mse
):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", device_map="cuda", torch_dtype=model_dtype
    )
    untie_word_embeddings(model)

    state = State(model=model)
    modifier = modifier(
        transform_type="random-hadamard",
        precision=precision,
        transform_block_size=transform_block_size,
    )

    input = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}
    with torch.no_grad():
        true_output = model(**input)

    modifier.on_initialize(state)
    modifier.on_start(state, None)

    with torch.no_grad():
        output = model(**input)

    assert torch.nn.MSELoss()(output.logits, true_output.logits) <= exp_mse
