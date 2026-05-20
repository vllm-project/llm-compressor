def test_llama3_2_vision_example_regex_matching():
    """Test that regex patterns in llama3.2_vision_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Multi-modal projector patterns are properly ignored
    - Vision model patterns are properly ignored
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import MllamaForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    with skip_weights_download(MllamaForConditionalGeneration):
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    ignore_patterns = [
        "re:.*lm_head",
        "re:.*multi_modal_projector.*",
        "re:.*vision_model.*",
    ]

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

    # Test multi_modal_projector is properly ignored
    projector_matches = [
        n for n, _ in linears_with_ignore if "multi_modal_projector" in n
    ]
    assert (
        len(projector_matches) == 0
    ), f"multi_modal_projector should be ignored, found {len(projector_matches)}"

    # Test vision_model is properly ignored
    vision_matches = [n for n, _ in linears_with_ignore if "vision_model" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_model should be ignored, found {len(vision_matches)}"
