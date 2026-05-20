from compressed_tensors.utils import match_named_modules
from transformers import MllamaForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_mllama_example_regex_matching():
    """Test that regex patterns in mllama_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - multi_modal_projector components are properly ignored
    - vision_model components are properly ignored
    """
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    with skip_weights_download(MllamaForConditionalGeneration):
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that multi_modal_projector is ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*multi_modal_projector.*"])
    )
    projector_matches = [n for n, _ in matches if "multi_modal_projector" in n]
    assert len(projector_matches) == 0, (
        f"multi_modal_projector should be ignored, but found "
        f"{len(projector_matches)} matches"
    )

    # Test that vision_model is properly ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*vision_model.*"])
    )
    vision_matches = [n for n, _ in matches if "vision_model" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_model should be ignored, but found {len(vision_matches)} matches"

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
