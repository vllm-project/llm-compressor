"""Test regex pattern matching for Idefics3 example."""

from compressed_tensors.utils import match_named_modules
from transformers import Idefics3ForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_idefics3_example_regex_matching():
    """Test that regex patterns in idefics3_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - vision_model modules are properly ignored
    - connector modules are properly ignored
    - Linear modules in the language model are still targeted
    """
    model_id = "HuggingFaceM4/Idefics3-8B-Llama3"

    with skip_weights_download(Idefics3ForConditionalGeneration):
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head exists
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head module, got {len(matches)}"

    # Test vision_model pattern
    matches = list(match_named_modules(model, ["re:model.vision_model.*"], ignore=[]))
    vision_model_count = len(matches)
    assert (
        vision_model_count > 0
    ), f"Expected vision_model modules, got {vision_model_count}"

    # Test connector pattern
    matches = list(match_named_modules(model, ["re:model.connector.*"], ignore=[]))
    connector_count = len(matches)
    assert connector_count > 0, f"Expected connector modules, got {connector_count}"

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
                    "re:.*lm_head",
                    "re:model.vision_model.*",
                    "re:model.connector.*",
                ],
            )
        )
    )
    assert ignored_linear_count < all_linear_count, (
        f"Expected ignored count ({ignored_linear_count}) to be less than "
        f"total count ({all_linear_count})"
    )

    # Verify that vision_model modules are excluded when ignored
    language_linear_matches = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=["re:.*lm_head", "re:model.vision_model.*", "re:model.connector.*"],
        )
    )
    vision_in_language = [
        name for name, _ in language_linear_matches if "vision_model" in name
    ]
    assert len(vision_in_language) == 0, (
        f"Expected no vision_model modules in language model targets, "
        f"found {len(vision_in_language)}: {vision_in_language}"
    )

    # Verify that connector modules are excluded when ignored
    connector_in_language = [
        name for name, _ in language_linear_matches if "connector" in name
    ]
    assert len(connector_in_language) == 0, (
        f"Expected no connector modules in language model targets, "
        f"found {len(connector_in_language)}: {connector_in_language}"
    )
