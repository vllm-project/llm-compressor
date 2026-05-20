from compressed_tensors.utils import match_named_modules
from transformers import AutoModelForCausalLM

from llmcompressor.utils.dev import skip_weights_download


def test_phi3_vision_example_regex_matching():
    """Test that regex patterns in phi3_vision_example match expected modules.

    This test validates that:
    - lm_head is properly ignored (should not be matched)
    - model.vision_embed_tokens components are properly ignored
    """
    model_id = "microsoft/Phi-3-vision-128k-instruct"

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="meta",
            trust_remote_code=True,
            _attn_implementation="eager",
        )

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if n == "lm_head"]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, but found {len(lm_head_matches)} matches"

    # Test that model.vision_embed_tokens is properly ignored
    matches = list(
        match_named_modules(
            model, ["Linear"], ignore=["re:model.vision_embed_tokens.*"]
        )
    )
    vision_matches = [n for n, _ in matches if "model.vision_embed_tokens" in n]
    assert len(vision_matches) == 0, (
        f"model.vision_embed_tokens should be ignored, but found "
        f"{len(vision_matches)} matches"
    )

    # Test that lm_head can be matched when not ignored
    matches = list(match_named_modules(model, ["lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"
