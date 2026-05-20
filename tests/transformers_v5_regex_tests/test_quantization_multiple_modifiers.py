def test_quantization_multiple_modifiers_regex_matching():
    """Test that regex patterns in quantization_multiple_modifiers match expected modules.

    This test validates that:
    - self_attn projection layers (k_proj, o_proj, q_proj, v_proj) are matched for GPTQ
    - mlp layers (down_proj, gate_proj, up_proj) are matched for AWQ
    - The two modifier target groups don't overlap
    - AWQ mappings reference valid source and target layers
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test self_attn projection patterns (GPTQ targets)
    gptq_pattern = r"re:.*self_attn\.(k|q|o|v)_proj$"
    gptq_matches = list(match_named_modules(model, [gptq_pattern], ignore=[]))
    assert (
        len(gptq_matches) == 128
    ), f"Expected 128 attention projection layers (32 layers * 4 projs), got {len(gptq_matches)}"

    # Test mlp patterns (AWQ targets)
    awq_pattern = r"re:.*mlp\.(down|gate|up)_proj$"
    awq_matches = list(match_named_modules(model, [awq_pattern], ignore=[]))
    assert (
        len(awq_matches) == 96
    ), f"Expected 96 mlp layers (32 layers * 3 projs), got {len(awq_matches)}"

    # Verify no overlap between GPTQ and AWQ targets
    gptq_names = {n for n, _ in gptq_matches}
    awq_names = {n for n, _ in awq_matches}
    overlap = gptq_names & awq_names
    assert (
        len(overlap) == 0
    ), f"GPTQ and AWQ targets should not overlap, but found {len(overlap)} overlapping modules"

    # Test AWQ mapping source: post_attention_layernorm
    layernorm_pattern = "re:.*post_attention_layernorm$"
    layernorm_matches = list(match_named_modules(model, [layernorm_pattern], ignore=[]))
    assert (
        len(layernorm_matches) == 32
    ), f"Expected 32 post_attention_layernorm, got {len(layernorm_matches)}"

    # Test AWQ mapping targets: gate_proj and up_proj
    gate_up_pattern = ["re:.*gate_proj$", "re:.*up_proj$"]
    gate_up_matches = list(match_named_modules(model, gate_up_pattern, ignore=[]))
    assert (
        len(gate_up_matches) == 64
    ), f"Expected 64 gate/up_proj layers (32 * 2), got {len(gate_up_matches)}"

    # Test AWQ second mapping: up_proj -> down_proj
    up_proj_pattern = "re:.*up_proj$"
    up_proj_matches = list(match_named_modules(model, [up_proj_pattern], ignore=[]))
    assert (
        len(up_proj_matches) == 32
    ), f"Expected 32 up_proj layers, got {len(up_proj_matches)}"

    down_proj_pattern = "re:.*down_proj$"
    down_proj_matches = list(match_named_modules(model, [down_proj_pattern], ignore=[]))
    assert (
        len(down_proj_matches) == 32
    ), f"Expected 32 down_proj layers, got {len(down_proj_matches)}"
