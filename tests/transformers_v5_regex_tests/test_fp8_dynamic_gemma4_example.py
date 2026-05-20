def test_fp8_dynamic_gemma4_example_regex_matching():
    """Test that regex patterns in quantization_w8a8_fp8/gemma4_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Vision tower patterns are properly ignored
    - Embed patterns are properly ignored
    - Router patterns are properly ignored
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Gemma4ForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "google/gemma-4-26B-A4B-it"

    with skip_weights_download(Gemma4ForConditionalGeneration):
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    ignore_patterns = [
        "lm_head",
        "re:.*embed.*",
        "re:.*router",
        "re:.*vision_tower.*",
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

    # Test vision_tower is properly ignored
    vision_matches = [n for n, _ in linears_with_ignore if "vision_tower" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_tower should be ignored, found {len(vision_matches)}"

    # Test embed is properly ignored
    embed_matches = [n for n, _ in linears_with_ignore if "embed" in n]
    assert (
        len(embed_matches) == 0
    ), f"embed layers should be ignored, found {len(embed_matches)}"

    # Test router is properly ignored
    router_matches = [n for n, _ in linears_with_ignore if "router" in n]
    assert (
        len(router_matches) == 0
    ), f"router layers should be ignored, found {len(router_matches)}"
