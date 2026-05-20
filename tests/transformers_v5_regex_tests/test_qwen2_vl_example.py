from compressed_tensors.utils import match_named_modules
from transformers import Qwen2VLForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_qwen2_vl_example_regex_matching():
    """Test that regex patterns in qwen2_vl_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - visual components are properly ignored
    - model.visual components are properly ignored
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    with skip_weights_download(Qwen2VLForConditionalGeneration):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if n == "lm_head"]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that visual components are properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:visual.*"]))
    visual_matches = [n for n, _ in matches if n.startswith("visual")]
    assert (
        len(visual_matches) == 0
    ), f"visual.* should be ignored, but found {len(visual_matches)} matches"

    # Test that model.visual components are properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:model.visual.*"]))
    model_visual_matches = [n for n, _ in matches if "model.visual" in n]
    assert len(model_visual_matches) == 0, (
        f"model.visual.* should be ignored, but found "
        f"{len(model_visual_matches)} matches"
    )

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
