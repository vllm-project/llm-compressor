def test_mixtral_example_regex_matching():
    """Test that regex patterns in mixtral_example match expected modules.

    Validates:
    - lm_head is properly ignored
    - MoE gate layers are properly ignored
    - Other Linear layers are matched
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    with torch.device("meta"), load_quantizable_moe(), skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    ignores = ["lm_head", "re:.*mlp.gate"]

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignores))
    lm_head_matches = [name for name, _ in matches if "lm_head" in name]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test that MoE gate layers are ignored
    gate_matches = [name for name, _ in matches if "mlp.gate" in name]
    assert (
        len(gate_matches) == 0
    ), f"MoE gate should be ignored, found {len(gate_matches)} matches"

    # Test that expert layers are matched
    expert_matches = [name for name, _ in matches if "mlp.experts" in name]
    assert (
        len(expert_matches) > 0
    ), f"Expected MoE expert matches, got {len(expert_matches)}"

    # Test that attention layers are matched
    attn_matches = [name for name, _ in matches if "self_attn" in name]
    assert (
        len(attn_matches) > 0
    ), f"Expected attention layer matches, got {len(attn_matches)}"
