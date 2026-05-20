def test_fp8_block_example_regex_matching():
    """Test that regex patterns in fp8_block_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - MLP gate layers matching 're:.*mlp.gate$' are properly ignored
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-30B-A3B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is properly ignored
    all_linears = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in all_linears if "lm_head" in n]
    assert (
        len(lm_head_matches) == 1
    ), f"Expected 1 lm_head module, got {len(lm_head_matches)}"

    linears_no_lm_head = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head"])
    )
    lm_head_should_be_empty = [n for n, _ in linears_no_lm_head if "lm_head" in n]
    assert (
        len(lm_head_should_be_empty) == 0
    ), f"lm_head should be ignored, found {len(lm_head_should_be_empty)}"

    # Test mlp.gate regex pattern is properly ignored
    linears_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head", "re:.*mlp.gate$"])
    )
    gate_matches = [n for n, _ in linears_with_ignore if n.endswith("mlp.gate")]
    assert (
        len(gate_matches) == 0
    ), f"mlp.gate layers should be ignored, found {len(gate_matches)}"
