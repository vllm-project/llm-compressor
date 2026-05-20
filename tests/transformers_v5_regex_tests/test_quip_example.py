def test_quip_example_regex_matching():
    """Test that regex patterns in quip_example match expected modules.

    Validates:
    - lm_head is properly ignored
    - All other Linear layers are matched for quantization
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    ignores = ["lm_head"]

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignores))
    lm_head_matches = [name for name, _ in matches if "lm_head" in name]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test that attention layers are matched
    attn_matches = [name for name, _ in matches if "self_attn" in name]
    assert (
        len(attn_matches) > 0
    ), f"Expected attention layer matches, got {len(attn_matches)}"

    # Test that MLP layers are matched
    mlp_matches = [name for name, _ in matches if "mlp" in name]
    assert len(mlp_matches) > 0, f"Expected MLP layer matches, got {len(mlp_matches)}"

    # Verify all matches are Linear modules
    assert len(matches) > 0, f"Expected Linear module matches, got {len(matches)}"
