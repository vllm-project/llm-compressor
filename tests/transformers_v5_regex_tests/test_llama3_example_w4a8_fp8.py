def test_llama3_example_w4a8_fp8_regex_matching():
    """Test that regex patterns in llama3_example (W4AFP8) match expected modules.

    Recipe targets: Linear (via GPTQModifier with W4AFP8 scheme)
    Recipe ignore: lm_head

    This test validates that lm_head is properly excluded from Linear targets.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head exists and matches when not ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test lm_head is properly ignored when specified
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"
