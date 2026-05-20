def test_qwen2vl_example_regex_matching():
    """Test that regex patterns in qwen2vl_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (1 module matches the pattern)
    - visual modules are properly ignored (0 visual modules should remain after filtering)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen2VLForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    with skip_weights_download(Qwen2VLForConditionalGeneration):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is matched by ignore pattern
    lm_head_matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test visual modules are matched by ignore pattern
    visual_matches = list(match_named_modules(model, ["re:.*visual.*"], ignore=[]))
    assert (
        len(visual_matches) > 0
    ), f"Expected visual modules to exist, got {len(visual_matches)}"

    # Test that Linear modules exclude visual and lm_head when ignored
    all_linear = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*lm_head", "re:.*visual.*"])
    )

    # Filtered should have fewer modules than all
    assert len(filtered_linear) < len(
        all_linear
    ), "Ignore patterns should reduce module count"

    # Check no visual or lm_head in filtered results
    filtered_names = [name for name, _ in filtered_linear]
    visual_in_filtered = [n for n in filtered_names if "visual" in n]
    lm_head_in_filtered = [n for n in filtered_names if "lm_head" in n]

    assert (
        len(visual_in_filtered) == 0
    ), f"Visual modules should be ignored, found: {visual_in_filtered}"
    assert (
        len(lm_head_in_filtered) == 0
    ), f"lm_head should be ignored, found: {lm_head_in_filtered}"
