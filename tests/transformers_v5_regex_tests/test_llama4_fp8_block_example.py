def test_llama4_fp8_block_example_regex_matching():
    """Test that regex patterns in llama4_fp8_block_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Self-attention patterns are properly ignored
    - Router patterns are properly ignored
    - Vision model patterns are properly ignored
    - Multi-modal projector patterns are properly ignored
    - Llama4TextAttention class targets are properly ignored
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    ignore_patterns = [
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
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

    # Test self_attn is properly ignored
    self_attn_matches = [n for n, _ in linears_with_ignore if "self_attn" in n]
    assert (
        len(self_attn_matches) == 0
    ), f"self_attn should be ignored, found {len(self_attn_matches)}"

    # Test router is properly ignored
    router_matches = [n for n, _ in linears_with_ignore if "router" in n]
    assert (
        len(router_matches) == 0
    ), f"router should be ignored, found {len(router_matches)}"

    # Test vision_model is properly ignored
    vision_matches = [n for n, _ in linears_with_ignore if "vision_model" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_model should be ignored, found {len(vision_matches)}"

    # Test multi_modal_projector is properly ignored
    projector_matches = [
        n for n, _ in linears_with_ignore if "multi_modal_projector" in n
    ]
    assert (
        len(projector_matches) == 0
    ), f"multi_modal_projector should be ignored, found {len(projector_matches)}"
