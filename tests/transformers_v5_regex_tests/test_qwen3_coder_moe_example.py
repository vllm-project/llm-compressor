"""
Test regex pattern matching for awq/qwen3_coder_moe_example.py

This test validates that the AWQModifier recipe correctly:
- Targets all Linear layers
- Ignores lm_head, mlp gate layers, and shared expert gate layers
"""

import pytest
from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from llmcompressor.utils.dev import skip_weights_download


@pytest.mark.unit
def test_qwen3_coder_moe_example_regex_matching():
    import torch

    """Test that regex patterns in qwen3_coder_moe_example match expected modules."""

    model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    with torch.device("meta"), load_quantizable_moe(), skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is correctly identified (should exist)
    lm_head_matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test mlp.gate pattern (should match some gates)
    mlp_gate_matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    assert len(mlp_gate_matches) > 0, "Expected to find mlp.gate layers"
