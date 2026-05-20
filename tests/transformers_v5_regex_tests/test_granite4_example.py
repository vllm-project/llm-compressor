def test_granite4_example_regex_matching():
    """Test that regex patterns in granite4_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - Router patterns are properly ignored when skip_router_only is True
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "ibm-granite/granite-4.0-tiny-preview"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test with skip_router_only=True configuration
    ignore_patterns = ["lm_head", "re:.*block_sparse_moe.router"]

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

    # Test block_sparse_moe.router is properly ignored
    router_matches = [
        n for n, _ in linears_with_ignore if "block_sparse_moe.router" in n
    ]
    assert (
        len(router_matches) == 0
    ), f"block_sparse_moe.router should be ignored, found {len(router_matches)}"
