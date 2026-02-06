import contextlib
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.minimax_m2 import CalibrationMiniMaxM2SparseMoeBlock
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence, requires_gpu

MiniMaxM2Config = pytest.importorskip(
    "transformers.models.minimax_m2.configuration_minimax_m2",
    reason="MiniMaxM2Config not available in this version of transformers",
).MiniMaxM2Config
MiniMaxM2SparseMoeBlock = pytest.importorskip(
    "transformers.models.minimax_m2.modeling_minimax_m2",
    reason="MiniMaxM2SparseMoeBlock not available in this version of transformers",
).MiniMaxM2SparseMoeBlock


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["hf-internal-testing/MiniMax-M2-Small"])
def test_calib_replace_minimax_m2_all_experts(model_stub):
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationMiniMaxM2SparseMoeBlock):
                moe_layer = module
                break

        assert moe_layer is not None

        num_experts = moe_layer.experts.num_experts
        seen_gate = [False for _ in range(num_experts)]
        seen_down = [False for _ in range(num_experts)]
        gate_ptrs = {
            moe_layer.experts.gate_up_proj[i].data_ptr(): i for i in range(num_experts)
        }
        down_ptrs = {
            moe_layer.experts.down_proj[i].data_ptr(): i for i in range(num_experts)
        }

        original_linear = F.linear

        def patched_linear(input, weight, *args, **kwargs):
            ptr = weight.data_ptr()
            if ptr in gate_ptrs:
                seen_gate[gate_ptrs[ptr]] = True
            if ptr in down_ptrs:
                seen_down[down_ptrs[ptr]] = True
            return original_linear(input, weight, *args, **kwargs)

        # Create dummy input tensor that simulates hidden_states
        hidden_dim = model.config.hidden_size
        batch, seq_len = 2, 8
        sample = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            dtype=moe_layer.experts.gate_up_proj.dtype,
            device=moe_layer.experts.gate_up_proj.device,
        )

        with torch.no_grad():
            F.linear = patched_linear  # patch only within this scope
            try:
                _ = moe_layer(sample)
            finally:
                F.linear = original_linear

        assert all(seen_gate), f"Not all experts were run (gate_up): {seen_gate}"
        assert all(seen_down), f"Not all experts were run (down_proj): {seen_down}"


@requires_gpu
def test_calib_minimax_m2_module():
    config = MiniMaxM2Config(
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        num_experts_per_tok=2,
        num_local_experts=4,
    )
    with torch.device("cuda"):
        original = MiniMaxM2SparseMoeBlock(config).eval()

    hidden_dim = config.hidden_size
    sample = torch.randn(2, 4, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationMiniMaxM2SparseMoeBlock(original, config, True)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.allclose(true_output, output, atol=1e-5)

    module = CalibrationMiniMaxM2SparseMoeBlock(original, config, False)
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.allclose(true_output, output, atol=1e-5)


def test_calib_minimax_m2_uses_upstream_experts_when_not_calibrating_all():
    config = MiniMaxM2Config(
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        num_experts_per_tok=2,
        num_local_experts=4,
    )
    original = MiniMaxM2SparseMoeBlock(config).eval()
    module = CalibrationMiniMaxM2SparseMoeBlock(original, config, False)

    sample = torch.randn(2, 4, config.hidden_size)

    with calibration_forward_context(original):
        true_output = original(sample)

    with mock.patch.object(
        module.experts, "forward", wraps=module.experts.forward
    ) as mocked_forward:
        with calibration_forward_context(module):
            output = module(sample)

    assert mocked_forward.call_count == 1
    assert torch.allclose(true_output, output, atol=1e-5)
