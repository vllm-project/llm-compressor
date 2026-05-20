from compressed_tensors.utils import match_named_modules
from transformers import Gemma3ForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_medgemma_example_regex_matching():
    """Test that regex patterns in medgemma_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - model.vision_tower components are properly ignored
    - model.multi_modal_projector components are properly ignored
    """
    model_id = "google/medgemma-27b-it"

    with skip_weights_download(Gemma3ForConditionalGeneration):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if n == "lm_head"]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that model.vision_tower is properly ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:model\\.vision_tower.*"])
    )
    vision_matches = [n for n, _ in matches if "model.vision_tower" in n]
    assert (
        len(vision_matches) == 0
    ), f"model.vision_tower should be ignored, but found {len(vision_matches)} matches"

    # Test that model.multi_modal_projector is ignored
    matches = list(
        match_named_modules(
            model, ["Linear"], ignore=["re:model\\.multi_modal_projector.*"]
        )
    )
    projector_matches = [n for n, _ in matches if "model.multi_modal_projector" in n]
    assert len(projector_matches) == 0, (
        f"model.multi_modal_projector should be ignored, but found "
        f"{len(projector_matches)} matches"
    )

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
