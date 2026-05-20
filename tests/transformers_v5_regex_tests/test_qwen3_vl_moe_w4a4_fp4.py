def test_qwen3_vl_moe_w4a4_fp4_regex_matching():
    """Test that regex patterns in qwen3_vl_moe_w4a4_fp4 match expected modules.

    Recipe targets: Linear
    Recipe ignore:
        - re:.*lm_head
        - re:visual.*
        - re:model.visual.*
        - re:.*mlp.gate$

    This test validates that visual components and MoE gates are properly ignored.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen3VLMoeForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    with skip_weights_download(Qwen3VLMoeForConditionalGeneration):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is properly ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test visual.* modules are properly ignored
    matches = list(match_named_modules(model, ["re:visual.*"], ignore=[]))
    visual_module_count = len(matches)

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:visual.*"]))
    visual_matches = [n for n, _ in matches if n.startswith("visual")]
    assert (
        len(visual_matches) == 0
    ), f"visual.* should be ignored, found {len(visual_matches)} matches out of {visual_module_count} total"

    # Test model.visual.* modules are properly ignored
    matches = list(match_named_modules(model, ["re:model.visual.*"], ignore=[]))
    model_visual_module_count = len(matches)

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:model.visual.*"]))
    model_visual_matches = [n for n, _ in matches if "model.visual" in n]
    assert (
        len(model_visual_matches) == 0
    ), f"model.visual.* should be ignored, found {len(model_visual_matches)} matches out of {model_visual_module_count} total"

    # Test mlp.gate modules are properly ignored
    matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    gate_module_count = len(matches)

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*mlp.gate$"]))
    gate_matches = [n for n, _ in matches if n.endswith("mlp.gate")]
    assert (
        len(gate_matches) == 0
    ), f"mlp.gate should be ignored, found {len(gate_matches)} matches out of {gate_module_count} total"
