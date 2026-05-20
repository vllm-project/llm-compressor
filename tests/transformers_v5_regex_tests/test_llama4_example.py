def test_llama4_example_regex_matching():
    """Test that regex patterns in llama4_example match expected modules.

    Recipe targets: Linear
    Recipe ignore:
        - re:.*lm_head
        - re:.*self_attn
        - re:.*router
        - re:.*vision_model.*
        - re:.*multi_modal_projector.*
        - Llama4TextAttention

    This test validates that vision components and multi-modal projector are properly ignored.
    """
    from compressed_tensors.utils import match_named_modules
    from transformers import Llama4ForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    with skip_weights_download(Llama4ForConditionalGeneration):
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id, device_map="meta"
        )

    # Test lm_head is properly ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test vision_model is properly ignored
    matches = list(match_named_modules(model, ["re:.*vision_model.*"], ignore=[]))
    vision_module_count = len(matches)

    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*vision_model.*"])
    )
    vision_matches = [n for n, _ in matches if "vision_model" in n]
    assert (
        len(vision_matches) == 0
    ), f"vision_model should be ignored, found {len(vision_matches)} matches out of {vision_module_count} total"

    # Test multi_modal_projector is properly ignored
    matches = list(
        match_named_modules(model, ["re:.*multi_modal_projector.*"], ignore=[])
    )
    projector_module_count = len(matches)

    matches = list(
        match_named_modules(model, ["Linear"], ignore=["re:.*multi_modal_projector.*"])
    )
    projector_matches = [n for n, _ in matches if "multi_modal_projector" in n]
    assert (
        len(projector_matches) == 0
    ), f"multi_modal_projector should be ignored, found {len(projector_matches)} matches out of {projector_module_count} total"

    # Test self_attn is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*self_attn"]))
    self_attn_matches = [n for n, _ in matches if "self_attn" in n]
    assert (
        len(self_attn_matches) == 0
    ), f"self_attn should be ignored, found {len(self_attn_matches)} matches"

    # Test router is properly ignored
    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*router"]))
    router_matches = [n for n, _ in matches if "router" in n]
    assert (
        len(router_matches) == 0
    ), f"router should be ignored, found {len(router_matches)} matches"
