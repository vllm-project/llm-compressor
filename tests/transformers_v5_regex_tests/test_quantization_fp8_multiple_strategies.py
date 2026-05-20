def test_quantization_fp8_multiple_strategies_regex_matching():
    """Test that regex patterns in quantization_fp8_multiple_strategies match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - self_attn projection layers (k_proj, o_proj, q_proj, v_proj) are matched correctly
    - mlp layers (down_proj, gate_proj, up_proj) are matched correctly
    - The two groups don't overlap
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is properly ignored
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in all_linear_matches if "lm_head" in n]
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    matches_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head"])
    )
    lm_head_ignored = [n for n, _ in matches_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_ignored) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_ignored)} matches"

    # Test self_attn projection patterns (group_0)
    attn_patterns = [
        "re:.*self_attn.k_proj.*",
        "re:.*self_attn.o_proj.*",
        "re:.*self_attn.q_proj.*",
        "re:.*self_attn.v_proj.*",
    ]
    attn_matches = list(match_named_modules(model, attn_patterns, ignore=["lm_head"]))
    assert (
        len(attn_matches) == 128
    ), f"Expected 128 attention projection layers (32 layers * 4 projs), got {len(attn_matches)}"

    # Test mlp patterns (group_1)
    mlp_patterns = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
    mlp_matches = list(match_named_modules(model, mlp_patterns, ignore=["lm_head"]))
    assert (
        len(mlp_matches) == 96
    ), f"Expected 96 mlp layers (32 layers * 3 projs), got {len(mlp_matches)}"

    # Verify no overlap between the two groups
    attn_names = {n for n, _ in attn_matches}
    mlp_names = {n for n, _ in mlp_matches}
    overlap = attn_names & mlp_names
    assert (
        len(overlap) == 0
    ), f"Groups should not overlap, but found {len(overlap)} overlapping modules"
