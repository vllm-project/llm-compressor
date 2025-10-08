import torch

from llmcompressor.modeling.qwen3_vl_moe import LinearQwen3VLMoeTextSparseMoeBlock
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_gpu


@requires_gpu
def test_calib_qwen3_vl_moe_module():
    from transformers import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )

    config = Qwen3VLMoeTextConfig()
    with torch.device("cuda"):
        original = Qwen3VLMoeTextSparseMoeBlock(config).eval()
        # these are initialized as empty / all 0s which results in output
        # from the experts being all 0 and incorrectly makes it seem like identical
        # outputs with our definition update to use a small random value
        original.experts.gate_up_proj.data.normal_(mean=0.0, std=0.01)
        original.experts.down_proj.data.normal_(mean=0.0, std=0.01)

    # Create dummy input tensor that simulates hidden_states
    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_output = original(sample)[0]

    module = LinearQwen3VLMoeTextSparseMoeBlock(
        config, original, calibrate_all_experts=True
    )
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)

    module = LinearQwen3VLMoeTextSparseMoeBlock(
        config, original, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        output = module(sample)[0]
        assert torch.allclose(true_output, output, atol=1e-6)
