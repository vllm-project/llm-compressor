def test_qwen3_vl_example_regex_matching():
    """Test that regex patterns in qwen3_vl_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - visual components (vision_tower, visual, etc.) are properly ignored
    - Linear layers excluding the ignored patterns are matched
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen3VLForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    with skip_weights_download(Qwen3VLForConditionalGeneration):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is properly ignored
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in all_linear_matches if "lm_head" in n]
    assert len(lm_head_matches) > 0, "lm_head should exist in the model"

    # Test lm_head is ignored with the recipe's ignore pattern
    matches_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*lm_head"])
    )
    lm_head_ignored = [n for n, _ in matches_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_ignored) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_ignored)} matches"

    # Test visual components are properly ignored
    visual_matches = [n for n, _ in all_linear_matches if "visual" in n]
    assert len(visual_matches) > 0, "visual components should exist in the model"

    matches_with_visual_ignore = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*visual.*"])
    )
    visual_ignored = [n for n, _ in matches_with_visual_ignore if "visual" in n]
    assert (
        len(visual_ignored) == 0
    ), f"visual components should be ignored, but found {len(visual_ignored)} matches"

    # Test both ignores together (as in the recipe)
    final_matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*lm_head", "re:.*visual.*"])
    )
    final_lm_head = [n for n, _ in final_matches if "lm_head" in n]
    final_visual = [n for n, _ in final_matches if "visual" in n]
    assert len(final_lm_head) == 0, "lm_head should be ignored in final matches"
    assert (
        len(final_visual) == 0
    ), "visual components should be ignored in final matches"
    assert (
        len(final_matches) > 0
    ), "Should still have some Linear layers after ignoring lm_head and visual"
