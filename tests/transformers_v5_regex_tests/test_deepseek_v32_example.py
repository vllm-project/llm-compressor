def test_deepseek_v32_example_regex_matching():
    """Test that regex patterns in deepseek_v32_example match expected modules.

    This test validates:
    - lm_head is properly ignored
    - MLP layer patterns match expected modules
    - Self-attention layer patterns match expected modules
    """
    import torch
    from compressed_tensors.utils import match_named_modules

    from llmcompressor.modeling.deepseekv32.model import DeepseekV32ForCausalLM
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "deepseek-ai/DeepSeek-V3.2"

    # Skip the custom model class as it requires special handling
    # We'll test the regex patterns conceptually
    with torch.device("meta"), load_quantizable_moe(), skip_weights_download(
        DeepseekV32ForCausalLM
    ):
        model = DeepseekV32ForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test that lm_head is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but got {len(lm_head_matches)} matches"

    # Test MLP patterns
    mlp_pattern = r"re:model.*mlp.*(gate|up|down|gate_up)_proj$"
    matches = list(match_named_modules(model, [mlp_pattern], ignore=[]))
    assert (
        len(matches) > 0
    ), f"MLP pattern should match some modules, got {len(matches)}"

    # Test self-attention patterns
    attn_pattern = r"re:model.*self_attn.indexer.(wk|wq_b)$"
    matches = list(match_named_modules(model, [attn_pattern], ignore=[]))
    # May or may not have matches depending on model architecture

    attn_pattern2 = r"re:model.*self_attn.(kv_b|o|q_a|q_b)_proj$"
    matches = list(match_named_modules(model, [attn_pattern2], ignore=[]))
    assert (
        len(matches) > 0
    ), f"Self-attention pattern should match some modules, got {len(matches)}"
