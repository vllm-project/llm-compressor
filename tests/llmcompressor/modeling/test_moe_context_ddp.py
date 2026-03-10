"""Integration tests for MoE calibration context in DDP mode.

Run with: torchrun --nproc_per_node=2 -m pytest
tests/llmcompressor/modeling/test_moe_context_ddp.py -v
"""

import pytest
import torch
from compressed_tensors.offload import load_offloaded_model
from torch import distributed as dist
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.deepseek_v3 import CalibrationDeepseekV3MoE
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download


@pytest.fixture(scope="module")
def ddp_environment():
    """Initialize DDP environment once for all tests."""
    if not dist.is_initialized():
        pytest.skip("DDP not initialized - run with torchrun")
    yield


@pytest.mark.parametrize("model_stub", ["unsloth/DeepSeek-R1-0528-BF16"])
def test_moe_context_ddp(ddp_environment, model_stub):
    """Verify MoE replacement works correctly in DDP mode."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with load_offloaded_model():
        with skip_weights_download():
            model = AutoModelForCausalLM.from_pretrained(
                model_stub, device_map="auto_offload"
            )

    with moe_calibration_context(model, calibrate_all_experts=True):
        # Count replaced modules
        replaced_count = sum(
            1
            for _, m in model.named_modules()
            if isinstance(m, CalibrationDeepseekV3MoE)
        )
        assert replaced_count > 0, f"Rank {rank}: No modules replaced"

        # Verify consistency across ranks
        count_tensor = torch.tensor([replaced_count], dtype=torch.long, device=next(model.parameters()).device)
        all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
        dist.all_gather(all_counts, count_tensor)
        assert all(
            c.item() == replaced_count for c in all_counts
        ), f"Rank {rank}: Inconsistent counts {[c.item() for c in all_counts]}"

    # Verify permanent modules remain (DeepSeekV3 is permanent)
    final_count = sum(
        1 for _, m in model.named_modules() if isinstance(m, CalibrationDeepseekV3MoE)
    )
    assert final_count == replaced_count
