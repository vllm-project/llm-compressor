def test_quantization_int4_int8_regex_matching():
    """Test that regex patterns in quantization_int4_int8 match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - down_proj layers are matched for int8 quantization (group_0)
    - self_attn and other mlp layers are matched for int4 quantization (group_1)
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

    # Test down_proj patterns (group_0 - int8)
    down_proj_matches = list(
        match_named_modules(model, ["re:.*down_proj.*"], ignore=["lm_head"])
    )
    assert (
        len(down_proj_matches) == 32
    ), f"Expected 32 down_proj layers, got {len(down_proj_matches)}"

    # Test other patterns (group_1 - int4)
    int4_patterns = [
        "re:.*self_attn.k_proj.*",
        "re:.*self_attn.o_proj.*",
        "re:.*self_attn.q_proj.*",
        "re:.*self_attn.v_proj.*",
        "re:.*gate_proj.*",
        "re:.*up_proj.*",
    ]
    int4_matches = list(match_named_modules(model, int4_patterns, ignore=["lm_head"]))
    assert (
        len(int4_matches) == 192
    ), f"Expected 192 layers (32 * (4 attn + 2 mlp)), got {len(int4_matches)}"

    # Verify no overlap between the two groups
    down_proj_names = {n for n, _ in down_proj_matches}
    int4_names = {n for n, _ in int4_matches}
    overlap = down_proj_names & int4_names
    assert (
        len(overlap) == 0
    ), f"Groups should not overlap, but found {len(overlap)} overlapping modules"

    # Verify total coverage (all Linear layers except lm_head)
    total_matched = len(down_proj_matches) + len(int4_matches)
    assert total_matched == len(
        matches_with_ignore
    ), f"Expected {len(matches_with_ignore)} total matches, got {total_matched}"
