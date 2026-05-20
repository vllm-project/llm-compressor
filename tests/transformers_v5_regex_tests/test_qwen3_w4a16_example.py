def test_qwen3_w4a16_example_regex_matching():
    """Test that regex patterns in quantization_w4a16/qwen3_example match expected modules.

    This test validates:
    - lm_head is properly ignored (should not match when in ignore list)
    - mlp.gate patterns are properly ignored via regex
    - Linear targets work correctly with ignore patterns
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-235B-A22B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head matching without ignore
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test lm_head is properly ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head", "re:.*mlp.gate$"])
    )
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)}"

    # Test mlp.gate regex ignore pattern
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head", "re:.*mlp.gate$"])
    )
    gate_matches = [n for n, _ in matches if n.endswith("mlp.gate")]
    assert (
        len(gate_matches) == 0
    ), f"mlp.gate modules should be ignored, found {len(gate_matches)}"

    # Verify some Linear modules are still matched
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head", "re:.*mlp.gate$"])
    )
    assert (
        len(matches) > 0
    ), "Expected some Linear modules to match after ignoring lm_head and mlp.gate"
