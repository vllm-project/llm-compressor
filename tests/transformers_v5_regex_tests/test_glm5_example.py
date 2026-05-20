def test_glm5_example_regex_matching():
    """Test that regex patterns in glm5_example match expected modules.

    Validates:
    - lm_head is properly ignored
    - Early dense layers (0-2) are properly ignored
    - Linear layers outside ignored patterns are matched
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "zai-org/GLM-5.1"

    with skip_weights_download(), load_quantizable_moe(), torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    moe_ignores = [
        "re:model.layers.0.*",
        "re:model.layers.1.*",
        "re:model.layers.2.*",
        "lm_head",
    ]

    # Test that lm_head is ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=moe_ignores))
    lm_head_matches = [name for name, _ in matches if "lm_head" in name]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test that early layers are ignored
    layer_0_matches = [name for name, _ in matches if "model.layers.0." in name]
    layer_1_matches = [name for name, _ in matches if "model.layers.1." in name]
    layer_2_matches = [name for name, _ in matches if "model.layers.2." in name]

    assert (
        len(layer_0_matches) == 0
    ), f"model.layers.0 should be ignored, found {len(layer_0_matches)} matches"
    assert (
        len(layer_1_matches) == 0
    ), f"model.layers.1 should be ignored, found {len(layer_1_matches)} matches"
    assert (
        len(layer_2_matches) == 0
    ), f"model.layers.2 should be ignored, found {len(layer_2_matches)} matches"

    # Test that later layers are matched
    layer_3_matches = [name for name, _ in matches if "model.layers.3." in name]
    assert (
        len(layer_3_matches) > 0
    ), f"Expected matches in model.layers.3, got {len(layer_3_matches)}"
