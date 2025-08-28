import contextlib
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.prepare import moe_calibration_context
from llmcompressor.modeling.qwen3_moe import (
    OriginalQwen3MoeSparseMoeBlock,
    Qwen3MoeConfig,
    Qwen3MoeSparseMoeBlock,
)
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["Qwen/Qwen3-30B-A3B"])
def test_calib_replace_qwen3moe_all_experts(model_stub):
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    # Qwen3MoE layer replacement is temporary within the context
    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(DisableQuantization(model))

        moe_calibration_context(model, stack, calibrate_all_experts=True)

        # Find one MoE layer
        moe_layer = None
        for name, module in model.named_modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock):
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
    config = Qwen3MoeConfig()
    with torch.device("cuda"):
        original = OriginalQwen3MoeSparseMoeBlock(config).eval()

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)[0]

    module = Qwen3MoeSparseMoeBlock(config, original, calibrate_all_experts=True)
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)

    module = Qwen3MoeSparseMoeBlock(config, original, calibrate_all_experts=False)
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)
