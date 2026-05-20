def test_phi35_fp8_kv_example_regex_matching():
    """Test that regex patterns in phi3.5_fp8_kv_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Linear layers excluding lm_head are matched
    - Phi-3.5 has fused QKV layers, so the structure is different from standard models
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "microsoft/Phi-3.5-mini-instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head exists in the model
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in all_linear_matches if "lm_head" in n]
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test lm_head is properly ignored with the recipe's ignore pattern
    matches_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head"])
    )
    lm_head_ignored = [n for n, _ in matches_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_ignored) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_ignored)} matches"

    # Verify we still have Linear layers after ignoring lm_head
    assert (
        len(matches_with_ignore) > 0
    ), "Should have Linear layers after ignoring lm_head"
    assert (
        len(matches_with_ignore) == len(all_linear_matches) - 1
    ), "Should have one less match after ignoring lm_head"
