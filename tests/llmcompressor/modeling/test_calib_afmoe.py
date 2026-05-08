import contextlib
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.afmoe import CalibrationAfmoeMoE
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu, requires_transformers_v4

pytestmark = requires_transformers_v4


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["arcee-ai/Trinity-Large-Thinking"])
def test_calib_replace_afmoe_all_experts(model_stub):
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub, trust_remote_code=True)

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        # Find an AfmoeMoE layer
        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationAfmoeMoE):
                moe_layer = module
                break

        assert moe_layer is not None

        num_experts = len(moe_layer.experts)
        expert_triggered = [False for _ in range(num_experts)]

        # Define the hook function
        def hook_fn(i, module, input, output):
            expert_triggered[i] = True

        # Attach hooks using functools.partial to bind each index
        for i, expert in enumerate(moe_layer.experts):
            expert.register_forward_hook(partial(hook_fn, i))

        # Create dummy input tensor that simulates hidden_states
        hidden_dim = model.config.hidden_size
        batch, seq_len = 4, 32
        sample = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)

        # Forward through the MoE layer directly
        with torch.no_grad():
            _ = moe_layer(sample)

        # Assert all experts are used
        assert all(
            expert_triggered
        ), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_afmoe_module():
    # Load a small model to get the AfmoeMoE class (loaded via trust_remote_code)
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            "arcee-ai/Trinity-Large-Thinking", trust_remote_code=True
        )

    # Extract the original AfmoeMoE class and config from the model
    original = None
    for module in model.modules():
        if type(module).__name__ == "AfmoeMoE":
            original = module
            break

    assert original is not None, "Could not find AfmoeMoE module in model"

    config = model.config

    # Move to GPU and initialize
    original = original.to("cuda")
    for param in original.parameters():
        param.data.normal_(mean=0.0, std=0.02)

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32

    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationAfmoeMoE(original, config, calibrate_all_experts=True)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output, output) < 0.1

    module = CalibrationAfmoeMoE(original, config, calibrate_all_experts=False)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output, output) < 0.1
