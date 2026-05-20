def test_nvfp4_gemma4_example_regex_matching():
    """Test that regex patterns in nvfp4/gemma4_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (not quantized)
    - Vision tower modules are properly ignored (re:.*vision_tower.*)
    - Audio tower modules are properly ignored (re:.*audio_tower.*)
    - Vision embedding projection modules are properly ignored (re:.*embed_vision.*)
    - Audio embedding projection modules are properly ignored (re:.*embed_audio.*)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForImageTextToText

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "google/gemma-4-E4B-it"

    with skip_weights_download(AutoModelForImageTextToText):
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="meta")

    # Test lm_head exists
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    # Test vision_tower is properly ignored
    all_linear_matches = list(match_named_modules(model, ["Linear"], ignore=[]))
    vision_tower_matches = [n for n, _ in all_linear_matches if "vision_tower" in n]

    ignored_matches = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=[
                "lm_head",
                "re:.*vision_tower.*",
                "re:.*audio_tower.*",
                "re:.*embed_vision.*",
                "re:.*embed_audio.*",
            ],
        )
    )
    ignored_vision_tower_matches = [
        n for n, _ in ignored_matches if "vision_tower" in n
    ]
    assert (
        len(ignored_vision_tower_matches) == 0
    ), f"Vision tower modules should be ignored, but found {len(ignored_vision_tower_matches)}"

    # Test audio_tower is properly ignored
    ignored_audio_tower_matches = [n for n, _ in ignored_matches if "audio_tower" in n]
    assert (
        len(ignored_audio_tower_matches) == 0
    ), f"Audio tower modules should be ignored, but found {len(ignored_audio_tower_matches)}"

    # Test embed_vision is properly ignored
    ignored_embed_vision_matches = [
        n for n, _ in ignored_matches if "embed_vision" in n
    ]
    assert (
        len(ignored_embed_vision_matches) == 0
    ), f"Vision embedding modules should be ignored, but found {len(ignored_embed_vision_matches)}"

    # Test embed_audio is properly ignored
    ignored_embed_audio_matches = [n for n, _ in ignored_matches if "embed_audio" in n]
    assert (
        len(ignored_embed_audio_matches) == 0
    ), f"Audio embedding modules should be ignored, but found {len(ignored_embed_audio_matches)}"
