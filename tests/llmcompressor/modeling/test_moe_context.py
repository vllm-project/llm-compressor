"""Unit tests for MoE calibration context in single-rank mode."""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE as OriginalDeepseekV3MoE,
)

from llmcompressor.modeling.deepseek_v3 import CalibrationDeepseekV3MoE
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download


@pytest.mark.parametrize("model_stub", ["unsloth/DeepSeek-R1-0528-BF16"])
def test_moe_context_replacement(model_stub):
    """Verify that MoE modules are correctly replaced and restored."""
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    original_count = sum(
        1 for _, m in model.named_modules() if isinstance(m, OriginalDeepseekV3MoE)
    )
    assert original_count > 0, "Model should have MoE modules"

    with moe_calibration_context(model, calibrate_all_experts=True):
        # Verify replacement
        calibration_count = sum(
            1
            for _, m in model.named_modules()
            if isinstance(m, CalibrationDeepseekV3MoE)
        )
        assert calibration_count == original_count

    # Verify permanent modules remain
    final_count = sum(
        1 for _, m in model.named_modules() if isinstance(m, CalibrationDeepseekV3MoE)
    )
    assert final_count == original_count


def test_moe_context_calibrate_flag():
    """Verify calibrate_all_experts flag is passed correctly."""
    config = DeepseekV3Config()
    with torch.device("cpu"):
        original = OriginalDeepseekV3MoE(config).eval()

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = original
            self.config = config

    for flag_value in [True, False]:
        model = TestModel()
        with moe_calibration_context(model, calibrate_all_experts=flag_value):
            for _, m in model.named_modules():
                if isinstance(m, CalibrationDeepseekV3MoE):
                    assert m.calibrate_all_experts is flag_value
