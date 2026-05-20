from compressed_tensors.utils import match_named_modules
from transformers import Mistral3ForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_mistral3_example_regex_matching():
    """Test that regex patterns in mistral3_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - vision_tower components are properly ignored
    - multi_modal_projector components are properly ignored
    """
    model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    with skip_weights_download(Mistral3ForConditionalGeneration):
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that vision_tower is properly ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*vision_tower.*"])
    )
    vision_matches = [n for n, _ in matches if "vision_tower" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_tower should be ignored, but found {len(vision_matches)} matches"

    # Test that multi_modal_projector is ignored
    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*multi_modal_projector.*"])
    )
    projector_matches = [n for n, _ in matches if "multi_modal_projector" in n]
    assert len(projector_matches) == 0, (
        f"multi_modal_projector should be ignored, but found "
        f"{len(projector_matches)} matches"
    )

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
