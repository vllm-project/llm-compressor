import pytest
import torch
from compressed_tensors.transform import apply_transform_config
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.transform.template.quip import QUIP


@pytest.mark.parametrize(
    "dtype,exp_max,exp_mse", [
        (torch.bfloat16, 1.1, 0.012),  # constructing and running transforms in float32 can improve to (~0.6562, ~0.0055)  # noqa: E501
        (torch.float32, 4e-4, 2e-9)
    ]
)
def test_apply_correctness(dtype, exp_max, exp_mse):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", device_map="cuda", torch_dtype=dtype
    )

    input = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}
    with torch.no_grad():
        true_output = model(**input)

    apply_transform_config(model, QUIP)
    with torch.no_grad():
        output = model(**input)

    assert torch.max(true_output.logits - output.logits) <= exp_max
    assert torch.nn.MSELoss()(output.logits, true_output.logits) <= exp_mse
