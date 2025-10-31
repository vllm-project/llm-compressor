import contextlib
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.modeling.qwen3_next_moe import CalibrationQwen3NextSparseMoeBlock
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["Qwen/Qwen3-Next-80B-A3B-Instruct"])
def test_calib_replace_qwen3moe_all_experts(model_stub):
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    # Qwen3MoE layer replacement is temporary within the context
    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(DisableQuantization(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        # Find one MoE layer
        moe_layer = None
        for name, module in model.named_modules():
            if isinstance(module, CalibrationQwen3NextSparseMoeBlock):
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
def test_calib_qwen3_moe_module():
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextSparseMoeBlock,
    )

    config = Qwen3NextConfig()
    with torch.device("cuda"):
        original = Qwen3NextSparseMoeBlock(config).eval()

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationQwen3NextSparseMoeBlock(
        original, config, calibrate_all_experts=True
    )

    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10

    module = CalibrationQwen3NextSparseMoeBlock(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10
