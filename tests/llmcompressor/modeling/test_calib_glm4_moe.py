import contextlib
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.glm4_moe import CalibrationGlm4MoeMoE
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu

Glm4MoeConfig = pytest.importorskip(
    "transformers.models.glm4_moe.configuration_glm4_moe",
    reason="Glm4MoeConfig not available in this version of transformers",
).Glm4MoeConfig
OriginalGlm4MoeMoE = pytest.importorskip(
    "transformers.models.glm4_moe.modeling_glm4_moe",
    reason="Glm4MoeMoE not available in this version of transformers",
).Glm4MoeMoE


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["zai-org/GLM-4.7"])
def test_calib_replace_glm4moe_all_experts(model_stub):
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub, trust_remote_code=True)

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        # Find a GLM4 MoE layer
        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationGlm4MoeMoE):
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
def test_calib_glm4moe_module():
    config = Glm4MoeConfig()
    with torch.device("cuda"):
        original = OriginalGlm4MoeMoE(config).eval()

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationGlm4MoeMoE(original, config, calibrate_all_experts=True)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.allclose(true_output, output, atol=1e-6)

    module = CalibrationGlm4MoeMoE(original, config, calibrate_all_experts=False)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.allclose(true_output, output, atol=1e-6)
