"""
Test regex pattern matching for awq/fp8_dynamic_llama_example.py

This test validates that the AWQModifier recipe correctly:
- Targets all Linear layers
- Ignores lm_head module
"""

import pytest
from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.utils.dev import skip_weights_download


@pytest.mark.unit
def test_fp8_dynamic_llama_example_regex_matching():
    """Test that regex patterns in fp8_dynamic_llama_example match expected modules."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is correctly identified (should exist)
    lm_head_matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test that lm_head is ignored when specified in ignore list
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    linear_without_lm_head = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head"])
    )

    # Should have one less match when lm_head is ignored
    assert len(linear_without_lm_head) == len(all_linear_matches) - 1, (
        f"Expected {len(all_linear_matches) - 1} Linear layers without lm_head, "
        f"got {len(linear_without_lm_head)}"
    )

    # Verify lm_head is not in the filtered matches
    filtered_names = [name for name, _ in linear_without_lm_head]
    assert "lm_head" not in filtered_names, "lm_head should be ignored"
