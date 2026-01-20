import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize(
    "modifier,model_dtype,precision,transform_block_size,exp_mse",
    [
        (QuIPModifier, torch.bfloat16, torch.bfloat16, None, 9e-6),  # 8.5831e-06
        (QuIPModifier, torch.bfloat16, torch.float32, 16, 6e-6),  # 5.4240e-06
        (QuIPModifier, torch.float32, torch.float32, 32, 7e-15),  # 5.8138e-15
        (SpinQuantModifier, torch.bfloat16, torch.bfloat16, None, 8e-3),  # 0.0079
        (SpinQuantModifier, torch.bfloat16, torch.float32, 16, 8e-3),  # 0.0079
        (SpinQuantModifier, torch.float32, torch.float32, 32, 8e-3),  # 0.0079
        (SpinQuantModifier, torch.float32, torch.float64, 64, 8e-3),  # 0.0079
    ],
)
def test_apply_correctness(
    modifier, model_dtype, precision, transform_block_size, exp_mse
):
    model = AutoModelForCausalLM.from_pretrained(
        "nm-testing/tinysmokellama-3.2", device_map="cuda", dtype=model_dtype
    )

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
