def test_llama3_attention_regex_matching():
    """Test that regex patterns in llama3_attention match expected modules.

    This test validates that:
    - LlamaAttention modules are matched correctly (target for attention quantization)
    - The count of LlamaAttention modules is as expected for Llama-3-8B
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test LlamaAttention is matched
    matches = list(match_named_modules(model, ["LlamaAttention"], ignore=[]))
    assert (
        len(matches) == 32
    ), f"Expected 32 LlamaAttention modules for Llama-3-8B, got {len(matches)}"

    # Verify all matches are LlamaAttention instances
    for name, module in matches:
        assert "self_attn" in name, f"Expected 'self_attn' in name, got {name}"
