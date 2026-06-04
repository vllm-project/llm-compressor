import contextlib
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.gpt_oss import (
    CalibrationLinearExperts,
    LinearExperts,
    convert_model_for_quantization_gptoss,
)
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["openai/gpt-oss-20b"])
def test_convert_model_for_quantization_gptoss(model_stub):
    """Test convert_model_for_quantization_gptoss correctly replaces modules."""
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            model_stub, trust_remote_code=True
        )

    # convert model for quantization
    convert_model_for_quantization_gptoss(model, calibrate_all_experts=True)

    # find CalibrationLinearExperts layer
    calib_layer = None
    for _, module in model.named_modules():
        if isinstance(module, CalibrationLinearExperts):
            calib_layer = module
            break

    assert (
        calib_layer is not None
    ), "No CalibrationLinearExperts found in model"
    assert calib_layer.calibrate_all_experts is True
    assert hasattr(calib_layer, "experts")
    assert len(calib_layer.experts) > 0


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["openai/gpt-oss-20b"])
def test_calib_replace_gptoss_all_experts(model_stub):
    """Test all experts are triggered when calibrate_all_experts=True."""
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            model_stub, trust_remote_code=True
        )

    # convert model with calibrate_all_experts enabled
    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        convert_model_for_quantization_gptoss(
            model, calibrate_all_experts=True
        )

        # find a CalibrationLinearExperts layer
        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationLinearExperts):
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
        hidden_dim = moe_layer.hidden_size
        batch, seq_len = 2, 16
        num_experts_plus_dummy = moe_layer.num_experts + 1

        # Create sample input
        sample = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)

        # Create router outputs (indices and weights)
        # Simulate router selecting 2 experts per token
        top_k = 2
        router_indices = torch.randint(
            0, moe_layer.num_experts, (batch, seq_len, top_k)
        )
        routing_weights = torch.randn(batch, seq_len, num_experts_plus_dummy)
        routing_weights = torch.softmax(routing_weights, dim=-1)

        # Forward through the MoE layer directly
        with torch.no_grad():
            _ = moe_layer(sample, router_indices, routing_weights)

        assert all(
            expert_triggered
        ), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_linear_experts_module():
    """Test correctness of CalibrationLinearExperts"""
    # Create a LinearExperts module
    hidden_size = 768
    intermediate_size = 2880
    num_experts = 8
    batch, seq_len = 2, 16
    top_k = 2

    with torch.device("cuda"):
        original = LinearExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        ).eval()

    # Create dummy input
    sample = torch.randn(batch, seq_len, hidden_size, device="cuda")

    # Create router outputs
    num_experts_plus_dummy = num_experts + 1
    router_indices = torch.randint(0, num_experts, (batch, seq_len, top_k)).to(
        "cuda"
    )
    routing_weights = torch.randn(batch, seq_len, num_experts_plus_dummy).to(
        "cuda"
    )
    routing_weights = torch.softmax(routing_weights, dim=-1)

    # Get original output
    with calibration_forward_context(original):
        true_output = original(sample, router_indices, routing_weights)

    # Test with calibrate_all_experts=True
    class MockConfig:
        pass

    config = MockConfig()
    module = CalibrationLinearExperts(
        original, config, calibrate_all_experts=True
    )
    with calibration_forward_context(module):
        output = module(sample, router_indices, routing_weights)
        assert torch.allclose(true_output, output, atol=1e-5)

    # Test with calibrate_all_experts=False
    module = CalibrationLinearExperts(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample, router_indices, routing_weights)
        assert torch.allclose(true_output, output, atol=1e-5)


@requires_gpu
def test_linear_experts_shape_normalization():
    """Test that _normalize_shapes work correctly."""
    hidden_size = 768
    intermediate_size = 2880
    num_experts = 8

    with torch.device("cuda"):
        module = LinearExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        ).eval()

    # Test 3D input
    batch, seq_len = 4, 32
    sample_3d = torch.randn(batch, seq_len, hidden_size, device="cuda")
    router_indices_3d = torch.randint(0, num_experts, (batch, seq_len, 2)).to(
        "cuda"
    )
    routing_weights_3d = torch.randn(batch, seq_len, num_experts + 1).to(
        "cuda"
    )

    x, indices, weights, B, H = module._normalize_shapes(
        sample_3d, router_indices_3d, routing_weights_3d
    )

    assert x.shape == (batch * seq_len, hidden_size)
    assert indices.shape == (batch * seq_len, 2)
    assert weights.shape == (batch * seq_len, num_experts + 1)
    assert B == batch
    assert H == hidden_size

    # Test 2D input (already flattened)
    tokens = batch * seq_len
    sample_2d = torch.randn(tokens, hidden_size, device="cuda")
    router_indices_2d = torch.randint(0, num_experts, (tokens, 2)).to("cuda")
    routing_weights_2d = torch.randn(tokens, num_experts + 1).to("cuda")

    x, indices, weights, B, H = module._normalize_shapes(
        sample_2d, router_indices_2d, routing_weights_2d
    )

    assert x.shape == (tokens, hidden_size)
    assert indices.shape == (tokens, 2)
    assert weights.shape == (tokens, num_experts + 1)
    assert B == 1
    assert H == hidden_size
