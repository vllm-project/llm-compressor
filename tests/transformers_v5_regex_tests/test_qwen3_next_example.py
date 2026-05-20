def test_qwen3_next_example_regex_matching():
    """Test that regex patterns in qwen3_next_example match expected modules.

    Recipe targets: Linear
    Recipe ignore:
        - lm_head
        - re:.*mlp.gate$
        - re:.*mlp.shared_expert_gate$
        - re:.*linear_attn.*

    This test validates that lm_head, MoE gates, and linear attention modules are properly ignored.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-Next-80B-A3B-Instruct"

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

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*mlp.gate$"]))
    gate_matches = [n for n, _ in matches if n.endswith("mlp.gate")]
    assert (
        len(gate_matches) == 0
    ), f"mlp.gate should be ignored, found {len(gate_matches)} matches out of {gate_module_count} total"

    # Test mlp.shared_expert_gate is properly ignored
    matches = list(
        match_named_modules(model, ["re:.*mlp.shared_expert_gate$"], ignore=[])
    )
    shared_gate_module_count = len(matches)

    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*mlp.shared_expert_gate$"])
    )
    shared_gate_matches = [
        n for n, _ in matches if n.endswith("mlp.shared_expert_gate")
    ]
    assert (
        len(shared_gate_matches) == 0
    ), f"mlp.shared_expert_gate should be ignored, found {len(shared_gate_matches)} matches out of {shared_gate_module_count} total"

    # Test linear_attn is properly ignored
    matches = list(match_named_modules(model, ["re:.*linear_attn.*"], ignore=[]))
    linear_attn_module_count = len(matches)

    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*linear_attn.*"])
    )
    linear_attn_matches = [n for n, _ in matches if "linear_attn" in n]
    assert (
        len(linear_attn_matches) == 0
    ), f"linear_attn should be ignored, found {len(linear_attn_matches)} matches out of {linear_attn_module_count} total"
