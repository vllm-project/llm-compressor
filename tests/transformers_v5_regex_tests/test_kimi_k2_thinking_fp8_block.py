def test_kimi_k2_thinking_fp8_block_regex_matching():
    """Test that regex patterns in kimi_k2_thinking_fp8_block match expected modules.

    This test validates:
    - Gate modules matching pattern "re:.*gate$" are properly ignored
    - lm_head is properly ignored
    - kv_a_proj_with_mqa is properly ignored
    - q_a_proj is properly ignored
    - embed_tokens is properly ignored
    - Linear layers match expected modules excluding ignored patterns
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "unsloth/Kimi-K2-Thinking-BF16"

    with torch.device("meta"), load_quantizable_moe(), skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Define the ignore patterns from the example
    ignore_patterns = [
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ]

    # Test without ignore patterns to see what would be matched
    all_matches = list(match_named_modules(model, ["Linear"], ignore=[]))

    # Test with ignore patterns
    matches_with_ignore = list(
        match_named_modules(model, ["Linear"], ignore=ignore_patterns)
    )

    # Test that gate modules are ignored
    gate_matches = [n for n, _ in matches_with_ignore if n.endswith("gate")]
    assert (
        len(gate_matches) == 0
    ), f"Gate modules should be ignored, but got {len(gate_matches)} matches"

    # Test that lm_head is ignored
    lm_head_matches = [n for n, _ in matches_with_ignore if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but got {len(lm_head_matches)} matches"

    # Test that kv_a_proj_with_mqa is ignored
    kv_a_matches = [n for n, _ in matches_with_ignore if "kv_a_proj_with_mqa" in n]
    assert (
        len(kv_a_matches) == 0
    ), f"kv_a_proj_with_mqa should be ignored, but got {len(kv_a_matches)} matches"

    # Test that q_a_proj is ignored
    q_a_matches = [n for n, _ in matches_with_ignore if "q_a_proj" in n]
    assert (
        len(q_a_matches) == 0
    ), f"q_a_proj should be ignored, but got {len(q_a_matches)} matches"

    # Test that embed_tokens is ignored
    embed_matches = [n for n, _ in matches_with_ignore if "embed_tokens" in n]
    assert (
        len(embed_matches) == 0
    ), f"embed_tokens should be ignored, but got {len(embed_matches)} matches"

    # Verify that some Linear modules are still matched after ignoring
    assert (
        len(matches_with_ignore) > 0
    ), f"Should match some Linear modules after ignoring patterns, got {len(matches_with_ignore)}"

    # Verify that we filtered out some modules
    assert len(matches_with_ignore) < len(
        all_matches
    ), "Ignore patterns should filter out some modules"
