def test_qwen_30b_a3b_regex_matching():
    """Test that regex patterns in qwen_30b_a3b match expected modules.

    Recipe targets: Linear
    Recipe ignore:
        - lm_head
        - re:.*mlp.gate$

    This test validates that lm_head and MoE gate modules are properly ignored.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-30B-A3B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is properly ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test mlp.gate modules are properly ignored
    matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    gate_module_count = len(matches)
    assert gate_module_count > 0, "Expected to find mlp.gate modules in MoE model"

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*mlp.gate$"]))
    gate_matches = [n for n, _ in matches if n.endswith("mlp.gate")]
    assert (
        len(gate_matches) == 0
    ), f"mlp.gate should be ignored, found {len(gate_matches)} matches out of {gate_module_count} total"
