def test_mxfp4_qwen3_example_regex_matching():
    """Test that regex patterns in mxfp4/qwen3_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (not quantized)
    - Linear layers excluding lm_head are targeted for quantization
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-30B-A3B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head exists
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test lm_head is properly ignored when targeting Linear
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    ignored_linear_matches = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head"])
    )
    assert (
        len(ignored_linear_matches) == len(all_linear_matches) - 1
    ), "Expected lm_head to be excluded from Linear matches"
