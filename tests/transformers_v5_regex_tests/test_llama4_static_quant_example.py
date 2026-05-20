def test_llama4_static_quant_example_regex_matching():
    """Test that regex patterns in quantization_w8a8_fp8/llama4_static_quant_example match expected modules.

    This test validates:
    - lm_head is properly ignored via regex pattern
    - router modules are properly ignored via regex
    - self_attn modules are properly ignored via regex
    - shared_expert modules are properly ignored via regex
    - multi_modal_projector is properly ignored via regex
    - vision_model is properly ignored via regex
    - Linear targets work correctly with all ignore patterns
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Llama4ForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    with skip_weights_download(Llama4ForConditionalGeneration):
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    ignore_patterns = [
        "re:.*lm_head",
        "re:.*router",
        "re:.*self_attn.*",
        "re:.*shared_expert.*",
        "re:multi_modal_projector.*",
        "re:vision_model",
    ]

    # Test lm_head matching without ignore
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test lm_head is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)}"

    # Test router modules are properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    router_matches = [n for n, _ in matches if "router" in n]
    assert (
        len(router_matches) == 0
    ), f"Router modules should be ignored, found {len(router_matches)}"

    # Test self_attn modules are properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    attn_matches = [n for n, _ in matches if "self_attn" in n]
    assert (
        len(attn_matches) == 0
    ), f"self_attn modules should be ignored, found {len(attn_matches)}"

    # Test shared_expert modules are properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    shared_expert_matches = [n for n, _ in matches if "shared_expert" in n]
    assert (
        len(shared_expert_matches) == 0
    ), f"shared_expert modules should be ignored, found {len(shared_expert_matches)}"

    # Test multi_modal_projector is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    projector_matches = [n for n, _ in matches if "multi_modal_projector" in n]
    assert (
        len(projector_matches) == 0
    ), f"multi_modal_projector should be ignored, found {len(projector_matches)}"

    # Test vision_model is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    vision_matches = [n for n, _ in matches if "vision_model" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_model should be ignored, found {len(vision_matches)}"

    # Verify some Linear modules are still matched (expert MLPs)
    matches = list(match_named_modules(model, ["Linear"], ignore=ignore_patterns))
    assert (
        len(matches) > 0
    ), "Expected some Linear modules to match after applying all ignore patterns"
