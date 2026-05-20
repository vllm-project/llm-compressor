def test_llama3_8b_w8a8_distributed_regex_matching():
    """Test that regex patterns in llama3_8b_w8a8_distributed example match expected modules.

    This test validates:
    - lm_head is properly ignored (should not match when ignored)
    - Linear layers match all expected linear modules in the model
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test that lm_head exists and would be matched by Linear target
    matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 1
    ), f"Expected 1 lm_head in Linear matches, got {len(lm_head_matches)}"

    # Test that lm_head is properly ignored when in ignore list
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but got {len(lm_head_matches)} matches"
