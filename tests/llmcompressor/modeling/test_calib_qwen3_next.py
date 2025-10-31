import torch

from llmcompressor.modeling.qwen3_next_moe import CalibrationQwen3NextSparseMoeBlock
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_gpu


@requires_gpu
def test_calib_qwen3_moe_module():
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
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
        #assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10

    module = CalibrationQwen3NextSparseMoeBlock(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample)
        #assert torch.nn.functional.mse_loss(true_output[0], output[0]) < 1e-10
        assert torch.nn.functional.mse_loss(true_output[1], output[1]) < 1e-10
