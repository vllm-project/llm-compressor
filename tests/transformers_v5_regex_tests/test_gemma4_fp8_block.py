def test_gemma4_fp8_block_regex_matching():
    """Test that regex patterns in gemma4_fp8_block match expected modules.

    This test validates:
    - Vision-related modules are properly ignored
    - lm_head is properly ignored
    - embed_tokens is properly ignored
    - Linear layers match expected modules excluding ignored patterns

    Note: This is a vision-language model (Gemma 4) that includes vision components
    that should not be quantized.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "google/gemma-4-31B-it"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Define the ignore patterns from the example
    ignore_patterns = ["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"]

    # Test that vision modules are ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    vision_matches = [n for n, _ in matches if "vision" in n]

    matches_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=ignore_patterns)
    )
    vision_matches_ignored = [n for n, _ in matches_with_ignore if "vision" in n]
    assert (
        len(vision_matches_ignored) == 0
    ), f"Vision modules should be ignored, but got {len(vision_matches_ignored)} matches"

    # Test that lm_head is ignored
    lm_head_matches = [n for n, _ in matches_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but got {len(lm_head_matches)} matches"

    # Test that embed_tokens is ignored
    embed_matches = [n for n, _ in matches_with_ignore if "embed_tokens" in n]
    assert (
        len(embed_matches) == 0
    ), f"embed_tokens should be ignored, but got {len(embed_matches)} matches"

    # Verify that some Linear modules are still matched (after ignoring vision/lm_head/embed)
    assert (
        len(matches_with_ignore) > 0
    ), f"Should match some Linear modules after ignoring patterns, got {len(matches_with_ignore)}"
