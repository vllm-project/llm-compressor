import os
import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import QuIPModifier
from tests.testing_utils import requires_gpu


@requires_gpu
# @pytest.mark.skipif(
#     (not os.getenv("HF_TOKEN")),
#     reason="Skipping tracing tests requiring gated model access",
# )
@pytest.mark.parametrize(
    "dtype,exp_mse",
    [
        (torch.bfloat16, 5e-3),
        (torch.float32, 5e-11),
    ],
)
def test_apply_correctness(dtype, exp_mse):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", device_map="cuda", torch_dtype=dtype
    )
    state = State(model=model)
    modifier = QuIPModifier(transform_type="random-hadamard")

    input = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}
    with torch.no_grad():
        true_output = model(**input)

    modifier.on_initialize(state)
    modifier.on_start(state, None)

    with torch.no_grad():
        output = model(**input)

    print(torch.nn.MSELoss()(output.logits, true_output.logits))
    assert torch.nn.MSELoss()(output.logits, true_output.logits) <= exp_mse
