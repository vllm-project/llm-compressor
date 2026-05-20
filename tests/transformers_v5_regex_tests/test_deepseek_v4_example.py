def test_deepseek_v4_example_regex_matching():
    """Test that regex patterns in deepseek_v4_example match expected modules.

    Validates:
    - Attention projection patterns match self_attn layers
    - MLP expert patterns match gate/up/down projections
    - No modules are incorrectly ignored (empty ignore list)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "inference-optimization/DSV4-tiny-empty"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test attention projection patterns
    attention_targets = [
        r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
        r"re:.*attn\.compressor\.indexer\.q_b_proj$",
    ]
    matches = list(match_named_modules(model, attention_targets, ignore=[]))
    assert len(matches) > 0, f"Expected attention projections, got {len(matches)}"

    # Verify all matches are attention layers
    for name, _ in matches:
        assert "attn" in name, f"Non-attention module matched: {name}"

    # Test expert MLP patterns
    expert_targets = [r"re:.*mlp\..*(gate|up|down)_proj$"]
    matches = list(match_named_modules(model, expert_targets, ignore=[]))
    assert len(matches) > 0, f"Expected MLP expert projections, got {len(matches)}"

    # Verify all matches are MLP layers
    for name, _ in matches:
        assert "mlp" in name, f"Non-MLP module matched: {name}"
        assert any(
            proj in name for proj in ["gate_proj", "up_proj", "down_proj"]
        ), f"Unexpected MLP module: {name}"
