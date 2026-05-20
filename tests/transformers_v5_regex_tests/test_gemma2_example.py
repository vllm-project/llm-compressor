def test_gemma2_example_regex_matching():
    """Test that regex patterns in gemma2_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (1 exact match)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "google/gemma-2-2b-it"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head exact match
    lm_head_matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test that Linear modules exclude lm_head when ignored
    all_linear = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))

    # Filtered should have fewer modules
    assert len(filtered_linear) < len(
        all_linear
    ), "Ignore patterns should reduce module count"
    assert (
        len(filtered_linear) == len(all_linear) - 1
    ), "Should exclude exactly 1 module (lm_head)"

    # Check lm_head not in filtered results
    filtered_names = [name for name, _ in filtered_linear]
    lm_head_in_filtered = [n for n in filtered_names if n == "lm_head"]

    assert len(lm_head_in_filtered) == 0, "lm_head should be ignored"
