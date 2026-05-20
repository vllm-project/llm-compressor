def test_qwen3_vl_moe_fp8_example_regex_matching():
    """Test that regex patterns in qwen3_vl_moe_fp8_example match expected modules.

    This test validates that:
    - lm_head is properly ignored
    - visual modules are properly ignored
    - model.visual modules are properly ignored
    - mlp.gate modules are properly ignored
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import Qwen3VLMoeForConditionalGeneration

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    with torch.device("meta"), load_quantizable_moe(), skip_weights_download(
        Qwen3VLMoeForConditionalGeneration
    ):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is matched by ignore pattern
    lm_head_matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(lm_head_matches) == 1, f"Expected 1 lm_head, got {len(lm_head_matches)}"

    # Test model.visual modules are matched by ignore pattern
    model_visual_matches = list(
        match_named_modules(model, ["re:model.visual.*"], ignore=[])
    )
    assert (
        len(model_visual_matches) > 0
    ), f"Expected model.visual modules to exist, got {len(model_visual_matches)}"

    # Test mlp.gate pattern exists
    mlp_gate_matches = list(match_named_modules(model, ["re:.*mlp.gate$"], ignore=[]))
    assert (
        len(mlp_gate_matches) > 0
    ), f"Expected mlp.gate modules to exist, got {len(mlp_gate_matches)}"

    # Test that Linear modules exclude all ignore patterns
    all_linear = list(match_named_modules(model, ["Linear"], ignore=[]))
    filtered_linear = list(
        match_named_modules(
            model,
            ["Linear"],
            ignore=[
                "re:.*lm_head",
                "re:visual.*",
                "re:model.visual.*",
                "re:.*mlp.gate$",
            ],
        )
    )

    # Filtered should have fewer modules
    assert len(filtered_linear) < len(
        all_linear
    ), "Ignore patterns should reduce module count"

    # Check none of the ignored patterns appear in filtered results
    filtered_names = [name for name, _ in filtered_linear]
    lm_head_in_filtered = [n for n in filtered_names if "lm_head" in n]
    visual_in_filtered = [n for n in filtered_names if n.startswith("visual")]
    model_visual_in_filtered = [
        n for n in filtered_names if n.startswith("model.visual")
    ]
    mlp_gate_in_filtered = [n for n in filtered_names if n.endswith("mlp.gate")]

    assert (
        len(lm_head_in_filtered) == 0
    ), f"lm_head should be ignored, found: {lm_head_in_filtered}"
    assert (
        len(visual_in_filtered) == 0
    ), f"visual modules should be ignored, found: {visual_in_filtered}"
    assert (
        len(model_visual_in_filtered) == 0
    ), f"model.visual modules should be ignored, found: {model_visual_in_filtered}"
    assert (
        len(mlp_gate_in_filtered) == 0
    ), f"mlp.gate should be ignored, found: {mlp_gate_in_filtered}"
