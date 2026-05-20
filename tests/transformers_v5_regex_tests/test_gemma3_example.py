"""Test regex pattern matching for Gemma 3 example."""

from compressed_tensors.utils import match_named_modules
from transformers import Gemma3ForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_gemma3_example_regex_matching():
    """Test that regex patterns in gemma3_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - vision_tower modules are properly ignored
    - multi_modal_projector modules are properly ignored
    - Linear modules in the language model are still targeted
    """
    model_id = "google/gemma-3-4b-it"

    with skip_weights_download(Gemma3ForConditionalGeneration):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head exists
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head module, got {len(matches)}"

    # Test vision_tower pattern
    matches = list(match_named_modules(model, [r"re:model\.vision_tower.*"], ignore=[]))
    vision_tower_count = len(matches)
    assert (
        vision_tower_count > 0
    ), f"Expected vision_tower modules, got {vision_tower_count}"

    # Test multi_modal_projector pattern
    matches = list(
        match_named_modules(model, [r"re:model\.multi_modal_projector.*"], ignore=[])
    )
    projector_count = len(matches)
    assert (
        projector_count > 0
    ), f"Expected multi_modal_projector modules, got {projector_count}"

    # Test that all Linear modules can be matched
    all_linear_count = len(list(match_named_modules(model, ["Linear"], ignore=[])))
    assert all_linear_count > 0, f"Expected Linear modules, got {all_linear_count}"

    # Test that ignored patterns reduce the Linear count
    ignored_linear_count = len(
        list(
            match_named_modules(
                model,
                ["Linear"],
                ignore=[
                    "lm_head",
                    r"re:model\.vision_tower.*",
                    r"re:model\.multi_modal_projector.*",
                ],
            )
        )
    )
    assert ignored_linear_count < all_linear_count, (
        f"Expected ignored count ({ignored_linear_count}) to be less than "
        f"total count ({all_linear_count})"
    )

    # Verify that vision_tower modules are excluded when ignored
    language_linear_matches = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=[
                "lm_head",
                r"re:model\.vision_tower.*",
                r"re:model\.multi_modal_projector.*",
            ],
        )
    )
    vision_in_language = [
        name for name, _ in language_linear_matches if "vision_tower" in name
    ]
    assert len(vision_in_language) == 0, (
        f"Expected no vision_tower modules in language model targets, "
        f"found {len(vision_in_language)}: {vision_in_language}"
    )

    # Verify that multi_modal_projector modules are excluded when ignored
    projector_in_language = [
        name for name, _ in language_linear_matches if "multi_modal_projector" in name
    ]
    assert len(projector_in_language) == 0, (
        f"Expected no multi_modal_projector modules in language model targets, "
        f"found {len(projector_in_language)}: {projector_in_language}"
    )
