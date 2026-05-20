def test_w4a4_fp4_gemma4_example_regex_matching():
    """Test that regex patterns in quantization_w4a4_fp4/gemma4_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (not quantized)
    - Embedding modules are properly ignored (re:.*embed.*)
    - Router modules are properly ignored (re:.*router)
    - Vision tower modules are properly ignored (re:.*vision_tower.*)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Gemma4ForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "google/gemma-4-26B-A4B-it"

    with skip_weights_download(Gemma4ForConditionalGeneration):
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head exists
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test all ignore patterns are properly applied
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))

    ignored_matches = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=[
                "lm_head",
                "re:.*embed.*",
                "re:.*router",
                "re:.*vision_tower.*",
            ],
        )
    )

    # Test vision_tower is properly ignored
    ignored_vision_tower_matches = [
        n for n, _ in ignored_matches if "vision_tower" in n
    ]
    assert (
        len(ignored_vision_tower_matches) == 0
    ), f"Vision tower modules should be ignored, but found {len(ignored_vision_tower_matches)}"

    # Test embed is properly ignored
    ignored_embed_matches = [n for n, _ in ignored_matches if "embed" in n]
    assert (
        len(ignored_embed_matches) == 0
    ), f"Embedding modules should be ignored, but found {len(ignored_embed_matches)}"

    # Test router is properly ignored
    ignored_router_matches = [n for n, _ in ignored_matches if "router" in n]
    assert (
        len(ignored_router_matches) == 0
    ), f"Router modules should be ignored, but found {len(ignored_router_matches)}"
