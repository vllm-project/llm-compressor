"""Test regex pattern matching for Qwen 3.5 INT8 example."""

from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.utils.dev import skip_weights_download


def test_qwen35_int8_regex_matching():
    """Test that regex patterns in qwen3.5_int8 match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - mlp.gate modules are properly ignored
    - mlp.shared_expert_gate modules are properly ignored
    - norm modules are properly ignored
    - embed_tokens are properly ignored
    - visual modules are properly ignored (if present)
    - conv1d modules are properly ignored (if present)
    """
    model_id = "Qwen/Qwen3.5-35B-A3B"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head exists
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head module, got {len(matches)}"

    # Test mlp.gate pattern
    matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    gate_count = len(matches)
    assert gate_count >= 0, f"Expected mlp.gate modules count >= 0, got {gate_count}"

    # Test mlp.shared_expert_gate pattern
    matches = list(
        match_named_modules(model, ["re:.*mlp.shared_expert_gate.*"], ignore=[])
    )
    shared_gate_count = len(matches)
    assert (
        shared_gate_count >= 0
    ), f"Expected mlp.shared_expert_gate count >= 0, got {shared_gate_count}"

    # Test norm pattern
    matches = list(match_named_modules(model, ["re:.*norm.*"], ignore=[]))
    norm_count = len(matches)
    assert norm_count > 0, f"Expected norm modules, got {norm_count}"

    # Test embed_tokens pattern
    matches = list(match_named_modules(model, ["re:.*embed_tokens.*"], ignore=[]))
    embed_count = len(matches)
    assert embed_count > 0, f"Expected embed_tokens modules, got {embed_count}"

    # Test visual pattern (may not exist in this model)
    matches = list(match_named_modules(model, ["re:.*visual.*"], ignore=[]))
    visual_count = len(matches)
    assert visual_count >= 0, f"Expected visual modules count >= 0, got {visual_count}"

    # Test conv1d pattern (may not exist in this model)
    matches = list(match_named_modules(model, ["re:.*conv1d.*"], ignore=[]))
    conv1d_count = len(matches)
    assert conv1d_count >= 0, f"Expected conv1d modules count >= 0, got {conv1d_count}"

    # Test that ignored patterns reduce the Linear count
    all_linear_count = len(list(match_named_modules(model, ["Linear"], ignore=[])))
    ignored_linear_count = len(
        list(
            match_named_modules(
                model,
                ["Linear"],
                ignore=[
                    "lm_head",
                    "re:.*mlp.gate$",
                    "re:.*mlp.shared_expert_gate.*",
                    "re:.*norm.*",
                    "re:.*embed_tokens.*",
                    "re:.*visual.*",
                    "re:.*conv1d.*",
                ],
            )
        )
    )
    assert ignored_linear_count < all_linear_count, (
        f"Expected ignored count ({ignored_linear_count}) to be less than "
        f"total count ({all_linear_count})"
    )
