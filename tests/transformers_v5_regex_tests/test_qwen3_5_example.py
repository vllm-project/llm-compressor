def test_mxfp4_qwen3_5_example_regex_matching():
    """Test that regex patterns in mxfp4/qwen3.5_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (not quantized)
    - Visual encoder modules are properly ignored (re:.*visual.*)
    - Linear attention modules are properly ignored (re:.*linear_attn.*)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen3_5ForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3.5-27B"

    with skip_weights_download(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head exists
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test visual encoder is properly ignored
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    visual_matches = [n for n, _ in all_linear_matches if "visual" in n]

    ignored_matches = list(
        match_named_modules(
            model, ["Linear"], ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"]
        )
    )
    ignored_visual_matches = [n for n, _ in ignored_matches if "visual" in n]
    assert (
        len(ignored_visual_matches) == 0
    ), f"Visual modules should be ignored, but found {len(ignored_visual_matches)}"

    # Test linear_attn is properly ignored
    linear_attn_matches = [n for n, _ in ignored_matches if "linear_attn" in n]
    assert (
        len(linear_attn_matches) == 0
    ), f"Linear attention modules should be ignored, but found {len(linear_attn_matches)}"
