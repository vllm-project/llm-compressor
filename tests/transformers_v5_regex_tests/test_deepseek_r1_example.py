def test_deepseek_r1_example_regex_matching():
    """Test that regex patterns in deepseek_r1_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (1 exact match)
    - mlp.gate modules are properly ignored
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoConfig, AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "unsloth/DeepSeek-R1-0528-BF16"

    # Load model without quantization config (as done in the example)
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "quantization_config"):
        del config.quantization_config

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="meta", config=config
        )

    # Test lm_head exact match
    lm_head_matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test mlp.gate pattern exists
    mlp_gate_matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    assert (
        len(mlp_gate_matches) > 0
    ), f"Expected mlp.gate modules to exist, got {len(mlp_gate_matches)}"

    # Test that Linear modules exclude both lm_head and mlp.gate when ignored
    all_linear = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear = list(
        match_named_modules(model, ["Linear"], ignore=["lm_head", "re:.*mlp.gate$"])
    )

    # Filtered should have fewer modules
    assert len(filtered_linear) < len(
        all_linear
    ), "Ignore patterns should reduce module count"

    # Check none of the ignored patterns appear in filtered results
    filtered_names = [name for name, _ in filtered_linear]
    lm_head_in_filtered = [n for n in filtered_names if n == "lm_head"]
    mlp_gate_in_filtered = [n for n in filtered_names if n.endswith("mlp.gate")]

    assert len(lm_head_in_filtered) == 0, "lm_head should be ignored"
    assert len(mlp_gate_in_filtered) == 0, "mlp.gate should be ignored"
