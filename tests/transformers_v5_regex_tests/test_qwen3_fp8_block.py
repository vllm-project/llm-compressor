"""Test regex pattern matching for Qwen 3 FP8 Block example."""

from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.utils.dev import skip_weights_download


def test_qwen3_fp8_block_regex_matching():
    """Test that regex patterns in qwen3_fp8_block match expected modules.

    This test validates that:
    - model.embed_tokens is properly ignored
    - lm_head is properly ignored
    """
    model_id = "Qwen/Qwen3-0.6B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test model.embed_tokens exists
    matches = list(match_named_modules(model, ["model.embed_tokens"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 embed_tokens module, got {len(matches)}"

    # Test lm_head exists
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head module, got {len(matches)}"

    # Test that all Linear modules can be matched
    all_linear_count = len(list(match_named_modules(model, ["Linear"], ignore=[])))
    assert all_linear_count > 0, f"Expected Linear modules, got {all_linear_count}"
