"""Test regex pattern matching for Qwen2 Audio example."""

from compressed_tensors.utils import match_named_modules
from transformers import Qwen2AudioForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_qwen2_audio_regex_matching():
    """Test that regex patterns in qwen2_audio match expected modules.

    This test validates that:
    - language_model.lm_head is properly ignored
    - audio_tower modules are properly ignored
    - Linear modules in the language model are still targeted
    """
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"

    with skip_weights_download(Qwen2AudioForConditionalGeneration):
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test language_model.lm_head exists
    matches = list(match_named_modules(model, ["language_model.lm_head"], ignore=[]))
    assert (
        len(matches) == 1
    ), f"Expected 1 language_model.lm_head module, got {len(matches)}"

    # Test audio_tower pattern
    matches = list(match_named_modules(model, ["re:audio_tower.*"], ignore=[]))
    audio_tower_count = len(matches)
    assert (
        audio_tower_count > 0
    ), f"Expected audio_tower modules, got {audio_tower_count}"

    # Test that all Linear modules can be matched
    all_linear_count = len(list(match_named_modules(model, ["Linear"], ignore=[])))
    assert all_linear_count > 0, f"Expected Linear modules, got {all_linear_count}"

    # Test that ignored patterns reduce the Linear count
    ignored_linear_count = len(
        list(
            match_named_modules(
                model, ["Linear"], ignore=["language_model.lm_head", "re:audio_tower.*"]
            )
        )
    )
    assert ignored_linear_count < all_linear_count, (
        f"Expected ignored count ({ignored_linear_count}) to be less than "
        f"total count ({all_linear_count})"
    )

    # Verify that audio_tower modules are excluded when ignored
    language_linear_matches = list(
        match_named_modules(
            model, ["Linear"], ignore=["language_model.lm_head", "re:audio_tower.*"]
        )
    )
    audio_in_language = [
        name for name, _ in language_linear_matches if "audio_tower" in name
    ]
    assert len(audio_in_language) == 0, (
        f"Expected no audio_tower modules in language model targets, "
        f"found {len(audio_in_language)}: {audio_in_language}"
    )
