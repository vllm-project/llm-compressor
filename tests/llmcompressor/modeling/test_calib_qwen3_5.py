import pytest
import torch

from llmcompressor.modeling.qwen3_5_moe import CalibrationQwen3_5MoeSparseMoeBlock
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_gpu

Qwen3_5MoeConfig = pytest.importorskip(
    "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
    reason="Qwen3_5MoeConfig not available in this version of transformers",
).Qwen3_5MoeConfig
Qwen3_5MoeSparseMoeBlock = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    reason="Qwen3_5MoeSparseMoeBlock not available in this version of transformers",
).Qwen3_5MoeSparseMoeBlock


@requires_gpu
def test_calib_qwen3_5_moe_module():
    config = Qwen3_5MoeConfig()
    with torch.device("cuda"):
        original = Qwen3_5MoeSparseMoeBlock(config.get_text_config()).eval()
        # 3D fused expert params are uninitialized (torch.empty),
        # initialize with small random values to get meaningful outputs
        original.experts.gate_up_proj.data.normal_(mean=0.0, std=0.02)
        original.experts.down_proj.data.normal_(mean=0.0, std=0.02)

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.get_text_config().hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=True
    )
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output, output) < 1e-10

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output, output) < 1e-10
