def test_trinity_large_nvfp4_regex_matching():
    """Test that regex patterns in trinity_large_nvfp4 match expected modules.

    Recipe targets: Linear
    Recipe ignore:
        - lm_head
        - re:.*self_attn.*
        - re:.*mlp.router.*

    This test validates that lm_head, self_attn, and MoE router modules are properly ignored.
    """
    import torch
    from compressed_tensors.utils import match_named_modules
    from transformers import AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.utils.dev import skip_weights_download

    model_id = "arcee-ai/Trinity-Large-Thinking"

    with torch.device("meta"), load_quantizable_moe(), skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    # Test lm_head is properly ignored
    matches = list(match_named_modules(model, ["re:.*lm_head"], ignore=[]))
    assert len(matches) == 1, f"Expected 1 lm_head, got {len(matches)}"

    matches = list(match_named_modules(model, ["Linear"], ignore=["lm_head"]))
    lm_head_matches = [n for n, _ in matches if "lm_head" in n]
    assert (
        len(lm_head_matches) == 0
    ), f"lm_head should be ignored, found {len(lm_head_matches)} matches"

    # Test self_attn is properly ignored
    matches = list(match_named_modules(model, ["re:.*self_attn.*"], ignore=[]))
    self_attn_module_count = len(matches)

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*self_attn.*"]))
    self_attn_matches = [n for n, _ in matches if "self_attn" in n]
    assert (
        len(self_attn_matches) == 0
    ), f"self_attn should be ignored, found {len(self_attn_matches)} matches out of {self_attn_module_count} total"

    # Test mlp.router modules are properly ignored
    matches = list(match_named_modules(model, ["re:.*mlp.router.*"], ignore=[]))
    router_module_count = len(matches)

    matches = list(match_named_modules(model, ["Linear"], ignore=["re:.*mlp.router.*"]))
    router_matches = [n for n, _ in matches if "mlp.router" in n]
    assert (
        len(router_matches) == 0
    ), f"mlp.router should be ignored, found {len(router_matches)} matches out of {router_module_count} total"
