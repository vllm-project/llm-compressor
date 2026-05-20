"""
Test regex pattern matching for awq/qwen3-vl-30b-a3b-Instruct-example.py

This test validates that the AWQModifier recipe correctly:
- Targets all Linear layers
- Ignores embed_tokens, layer norms, mlp gates, and visual components
"""

import pytest
from compressed_tensors.utils import match_named_modules
from transformers import Qwen3VLMoeForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


@pytest.mark.unit
def test_qwen3_vl_30b_a3b_instruct_example_regex_matching():
    """Test that regex patterns in qwen3-vl-30b-a3b-Instruct-example match expected modules."""
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    with skip_weights_download(Qwen3VLMoeForConditionalGeneration):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is correctly identified (should exist)
    lm_head_matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test embed_tokens pattern
    embed_tokens_matches = list(
        match_named_modules(model, ["re:.*embed_tokens"], ignore=[])
    )
    assert len(embed_tokens_matches) > 0, "Expected to find embed_tokens"

    # Test input_layernorm pattern
    input_layernorm_matches = list(
        match_named_modules(model, ["re:.*input_layernorm$"], ignore=[])
    )
    assert len(input_layernorm_matches) > 0, "Expected to find input_layernorm layers"

    # Test mlp.gate pattern
    mlp_gate_matches = list(match_named_modules(model, ["re:.*mlp[.]gate$"], ignore=[]))
    assert len(mlp_gate_matches) > 0, "Expected to find mlp.gate layers"

    # Test post_attention_layernorm pattern
    post_attn_layernorm_matches = list(
        match_named_modules(model, ["re:.*post_attention_layernorm$"], ignore=[])
    )
    assert (
        len(post_attn_layernorm_matches) > 0
    ), "Expected to find post_attention_layernorm layers"

    # Test norm pattern
    norm_matches = list(match_named_modules(model, ["re:.*norm$"], ignore=[]))
    assert len(norm_matches) > 0, "Expected to find norm layers"

    # Test visual pattern - should match visual components
    visual_matches = list(
        match_named_modules(model, ["re:model[.]visual.*"], ignore=[])
    )
    # Note: May be 0 if visual components are under different path

    # Test that Linear layers are properly filtered
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear_matches = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=[
                "re:.*embed_tokens",
                "re:.*input_layernorm$",
                "re:.*mlp[.]gate$",
                "re:.*post_attention_layernorm$",
                "re:.*norm$",
                "re:model[.]visual.*",
                "re:visual.*",
                "lm_head",
            ],
        )
    )

    # Verify that filtered results don't contain ignored patterns
    filtered_names = [name for name, _ in filtered_linear_matches]
    assert "lm_head" not in filtered_names, "lm_head should be ignored"
    assert not any(
        "embed_tokens" in name for name in filtered_names
    ), "embed_tokens should be ignored"
    assert not any(
        name.endswith("input_layernorm") for name in filtered_names
    ), "input_layernorm layers should be ignored"
    assert not any(
        name.endswith("mlp.gate") for name in filtered_names
    ), "mlp.gate layers should be ignored"
    assert not any(
        name.endswith("post_attention_layernorm") for name in filtered_names
    ), "post_attention_layernorm layers should be ignored"
    assert not any(
        "visual" in name for name in filtered_names
    ), "visual components should be ignored"

    # Ensure we still have some Linear layers after filtering
    assert (
        len(filtered_linear_matches) > 0
    ), "Should have some Linear layers after filtering"
    assert len(filtered_linear_matches) < len(
        all_linear_matches
    ), "Should have fewer Linear layers after filtering"
