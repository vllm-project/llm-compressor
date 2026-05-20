def test_qwen_2_5_vl_example_regex_matching():
    """Test that regex patterns in qwen_2_5_vl_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Visual patterns are properly ignored (both 're:visual.*' and 're:model.visual.*')
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen2_5_VLForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    with skip_weights_download(Qwen2_5_VLForConditionalGeneration):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    ignore_patterns = ["lm_head", "re:visual.*", "re:model.visual.*"]

    # Test lm_head is properly ignored
    all_linears = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in all_linears if "lm_head" in n]
    assert (
        len(lm_head_matches) == 1
    ), f"Expected 1 lm_head module, got {len(lm_head_matches)}"

    linears_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=ignore_patterns)
    )
    lm_head_should_be_empty = [n for n, _ in linears_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_should_be_empty) == 0
    ), f"lm_head should be ignored, found {len(lm_head_should_be_empty)}"

    # Test visual patterns are properly ignored
    visual_matches = [n for n, _ in linears_with_ignore if "visual" in n]
    assert (
        len(visual_matches) == 0
    ), f"visual layers should be ignored, found {len(visual_matches)}"
