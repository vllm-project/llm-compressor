from compressed_tensors.utils import match_named_modules
from transformers import Qwen3OmniMoeForConditionalGeneration

from llmcompressor.utils.dev import skip_weights_download


def test_qwen3_omni_example_regex_matching():
    """Test that regex patterns in qwen3_omni_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - visual components are properly ignored
    - code2wav components are properly ignored

    Note: This model uses the thinker submodule for quantization.
    """
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    with skip_weights_download(Qwen3OmniMoeForConditionalGeneration):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test against the thinker submodule (as done in the example)
    thinker = model.thinker

    # Test that lm_head is ignored
    matches = list(match_named_modules(thinker, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if n == "lm_head"]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that visual components are properly ignored
    matches = list(match_named_modules(thinker, ["Linear"], ignore=["re:.*visual.*"]))
    visual_matches = [n for n, _ in matches if "visual" in n]
    assert (
        len(visual_matches) == 0
    ), f".*visual.* should be ignored, but found {len(visual_matches)} matches"

    # Test that code2wav components are properly ignored
    matches = list(match_named_modules(thinker, ["Linear"], ignore=["re:.*code2wav.*"]))
    code2wav_matches = [n for n, _ in matches if "code2wav" in n]
    assert (
        len(code2wav_matches) == 0
    ), f".*code2wav.* should be ignored, but found {len(code2wav_matches)} matches"

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(thinker, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
