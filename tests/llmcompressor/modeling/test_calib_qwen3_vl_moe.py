import torch
from transformers import Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)

from llmcompressor.modeling.qwen3_vl_moe import CalibrateQwen3VLMoeTextSparseMoeBlock
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_gpu


@requires_gpu
def test_calib_qwen3_moe_module():
    config = Qwen3VLMoeTextConfig()
    with torch.device("cuda"):
        original = Qwen3VLMoeTextSparseMoeBlock(config).eval()

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrateQwen3VLMoeTextSparseMoeBlock.from_original(
        original, config, calibrate_all_experts=True
    )
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10

    module = CalibrateQwen3VLMoeTextSparseMoeBlock.from_original(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample)
        assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10
