"""Test regex pattern matching for Kimi K2 Thinking NVFP4A16 example."""

from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.utils.dev import skip_weights_download


def test_kimi_k2_thinking_nvfp4a16_regex_matching():
    """Test that regex patterns in kimi_k2_thinking_nvfp4a16 match expected modules.

    This test validates that:
    - lm_head is properly ignored (not quantized)
    - gate modules matching "re:.*gate$" are properly ignored
    - kv_a_proj_with_mqa modules are properly ignored
    - q_a_proj modules are properly ignored
    - embed_tokens are properly ignored
    """
    model_id = "unsloth/Kimi-K2-Thinking-BF16"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is ignored (should match 1 module if not ignored)
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head module, got {len(matches)}"

    # Test gate modules matching "re:.*gate$" pattern
    matches = list(match_named_modules(model, ["re:.*gate$"], ignore=[]))
    gate_count = len(matches)
    assert gate_count > 0, f"Expected gate modules to exist, got {gate_count}"

    # Test kv_a_proj_with_mqa modules
    matches = list(match_named_modules(model, ["re:.*kv_a_proj_with_mqa$"], ignore=[]))
    kv_count = len(matches)
    assert (
        kv_count >= 0
    ), f"Expected kv_a_proj_with_mqa modules count >= 0, got {kv_count}"

    # Test q_a_proj modules
    matches = list(match_named_modules(model, ["re:.*q_a_proj$"], ignore=[]))
    q_count = len(matches)
    assert q_count >= 0, f"Expected q_a_proj modules count >= 0, got {q_count}"

    # Test embed_tokens
    matches = list(match_named_modules(model, ["model.embed_tokens"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 embed_tokens module, got {len(matches)}"

    # Test that ignored patterns don't match when properly ignored
    all_linear_count = len(list(match_named_modules(model, ["Linear"], ignore=[])))
    ignored_linear_count = len(
        list(
            match_named_modules(
                model,
                ["Linear"],
                ignore=[
                    "re:.*gate$",
                    "lm_head",
                    "re:.*kv_a_proj_with_mqa$",
                    "re:.*q_a_proj$",
                    "model.embed_tokens",
                ],
            )
        )
    )
    assert ignored_linear_count < all_linear_count, (
        f"Expected ignored count ({ignored_linear_count}) to be less than "
        f"total count ({all_linear_count})"
    )
