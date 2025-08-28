import os
from functools import partial

import pytest
import torch
from transformers import Llama4ForConditionalGeneration

from llmcompressor.modeling.llama4 import (
    Llama4TextConfig,
    Llama4TextMoe,
    SequentialLlama4TextMoe,
)
from llmcompressor.modeling.prepare import replace_modules_for_calibration
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu


@requires_cadence("weekly")
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping tracing tests requiring gated model access",
)
@pytest.mark.parametrize("model_stub", ["meta-llama/Llama-4-Scout-17B-16E-Instruct"])
def test_calib_replace_llama4_moe_all_experts(model_stub):
    with skip_weights_download(Llama4ForConditionalGeneration):
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_stub, torch_dtype="auto"
        )

    replace_modules_for_calibration(model, calibrate_all_experts=True)

    # Find a Llama4 MoE layer
    moe_layer = None
    for module in model.modules():
        if isinstance(module, SequentialLlama4TextMoe):
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
    hidden_dim = model.config.text_config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, dtype=model.dtype)

    # Forward through the MoE layer directly
    with torch.no_grad():
        _ = moe_layer(sample)

    # Assert all experts are used
    assert all(expert_triggered), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_llama4_module():
    config = Llama4TextConfig()
    with torch.device("cuda"):
        original = Llama4TextMoe(config)

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)[0]

    module = SequentialLlama4TextMoe(config, original, calibrate_all_experts=True)
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)

    module = SequentialLlama4TextMoe(config, original, calibrate_all_experts=False)
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)
