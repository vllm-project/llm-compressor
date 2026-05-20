def test_whisper_example_regex_matching():
    """Test that regex patterns in whisper_example match expected modules.

    This test validates that:
    - proj_out is properly ignored (1 exact match)
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import WhisperForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "openai/whisper-large-v2"

    with skip_weights_download(WhisperForConditionalGeneration):
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test proj_out exact match
    proj_out_matches = list(match_named_modules(model, ["proj_out"], ignore=[]))
    assert (
        len(proj_out_matches) == 1
    ), f"Expected 1 proj_out, got {len(proj_out_matches)}"

    # Test that Linear modules exclude proj_out when ignored
    all_linear = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear = list(match_named_modules(model, ["Linear"], ignore=["proj_out"]))

    # Filtered should have fewer modules
    assert len(filtered_linear) < len(
        all_linear
    ), "Ignore patterns should reduce module count"
    assert (
        len(filtered_linear) == len(all_linear) - 1
    ), "Should exclude exactly 1 module (proj_out)"

    # Check proj_out not in filtered results
    filtered_names = [name for name, _ in filtered_linear]
    proj_out_in_filtered = [n for n in filtered_names if n == "proj_out"]

    assert len(proj_out_in_filtered) == 0, "proj_out should be ignored"
