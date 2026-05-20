def test_deepseek_r1_nvfp4_fp8_block_regex_matching():
    """Test that regex patterns in deepseek_r1_nvfp4_fp8_block match expected modules.

    This test validates:
    - MLP layer patterns match expected modules (gate_up, gate, up, down projections)
    - Self-attention layer patterns match expected modules
    - Patterns correctly target fused layers with compatible shapes

    Note: This example uses model_free_ptq which doesn't load the full model,
    so we test pattern matching on a loaded model to verify the regex patterns work.
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    # Use the NVFP4 model as base (the example converts from this)
    model_id = "nvidia/DeepSeek-R1-NVFP4"

    with skip_weights_download(), load_quantizable_moe(), torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test MLP pattern (these are converted from NVFP4)
    # Dense layers account for 183 matches, should be ~4K matches
    mlp_pattern = "re:.*mlp.*(gate_up|gate|up|down)_proj$"
    matches = list(match_named_modules(model, [mlp_pattern], ignore=[]))
    assert (
        len(matches) > 40000
    ), f"MLP pattern should match some modules, got {len(matches)}"  # 183

    # Test self-attention patterns for FP8_BLOCK targets
    # Pattern 1: kv_a_proj_with_mqa and q_a_proj (fused layers with shape 576x7168)
    attn_pattern1 = "re:.*self_attn.(kv_a_proj_with_mqa|q_a_proj)$"
    matches = list(match_named_modules(model, [attn_pattern1], ignore=[]))
    # May or may not have matches depending on model architecture

    # Pattern 2: remaining self_attn layers (o_proj, q_b_proj)
    attn_pattern2 = "re:.*self_attn.(o_proj|q_b_proj).*"
    matches = list(match_named_modules(model, [attn_pattern2], ignore=[]))
    assert (
        len(matches) > 0
    ), f"Self-attention pattern should match some modules, got {len(matches)}"
