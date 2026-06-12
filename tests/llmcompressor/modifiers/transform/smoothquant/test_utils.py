from unittest.mock import patch

import pytest

from llmcompressor.modifiers.transform.smoothquant.utils import (
    COHERE_SMOOTHQUANT_MAPPINGS,
    DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    DEFAULT_SMOOTHQUANT_MAPPINGS,
    MAPPINGS_REGISTRY,
    PHI3_VISION_SMOOTHQUANT_MAPPINGS,
    get_layer_mappings_from_architecture,
    handle_mapping_resolution_errors,
)

smoothquant_utils = "llmcompressor.modifiers.transform.smoothquant.utils"


@pytest.mark.unit
def test_handle_mapping_resolution_errors():
    README_LOCATION = (
        "https://github.com/vllm-project/llm-compressor/tree/main/"
        "src/llmcompressor/modifiers/transform/smoothquant"
    )

    @handle_mapping_resolution_errors
    def func_that_raises_exception():
        raise ValueError("An error occurred")

    with pytest.raises(RuntimeError) as excinfo:
        func_that_raises_exception()

    assert "Error resolving mappings for given architecture." in str(excinfo.value)
    assert "Please refer to the README at" in str(excinfo.value)
    assert README_LOCATION in str(excinfo.value)


@pytest.mark.unit
@patch(
    f"{smoothquant_utils}.MAPPINGS_REGISTRY", {"arch1": "mapping1", "arch2": "mapping2"}
)
@patch(f"{smoothquant_utils}.DEFAULT_SMOOTHQUANT_MAPPINGS", "default_mapping")
def test_get_layer_mappings_from_architecture():
    # Test when architecture is in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch1") == "mapping1"

    # Test when architecture is not in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch3") == "default_mapping"


@pytest.mark.unit
@pytest.mark.parametrize(
    "architecture,expected_mappings",
    [
        ("CohereForCausalLM", COHERE_SMOOTHQUANT_MAPPINGS),
        ("Cohere2ForCausalLM", COHERE_SMOOTHQUANT_MAPPINGS),
        ("Cohere2VisionForConditionalGeneration", COHERE_SMOOTHQUANT_MAPPINGS),
    ],
)
def test_new_architecture_mappings_resolve(architecture, expected_mappings):
    assert architecture in MAPPINGS_REGISTRY
    resolved = get_layer_mappings_from_architecture(architecture)
    assert resolved is expected_mappings
    assert resolved is not DEFAULT_SMOOTHQUANT_MAPPINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    "architecture,expected_mappings",
    [
        ("DeepseekV3ForCausalLM", DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS),
        ("Phi3ForCausalLM", PHI3_VISION_SMOOTHQUANT_MAPPINGS),
    ],
)
def test_specialized_architecture_overrides_default(architecture, expected_mappings):
    assert architecture in MAPPINGS_REGISTRY
    resolved = get_layer_mappings_from_architecture(architecture)
    assert resolved is expected_mappings
    assert resolved is not DEFAULT_SMOOTHQUANT_MAPPINGS


@pytest.mark.unit
def test_granite_for_causal_lm_in_registry():
    """Sanity: GraniteForCausalLM is wired to the standard default mapping."""
    assert "GraniteForCausalLM" in MAPPINGS_REGISTRY
    assert (
        get_layer_mappings_from_architecture("GraniteForCausalLM")
        is DEFAULT_SMOOTHQUANT_MAPPINGS
    )


@pytest.mark.unit
def test_granite_default_mapping_regex_matches_real_module_tree():
    """Construct GraniteForCausalLM on the meta device with a tiny config
    and assert DEFAULT_SMOOTHQUANT_MAPPINGS' regex matches expected
    attention + MLP modules per layer. No HF Hub downloads, no weight
    allocation. Validates the registry entry added in this PR.
    """
    import re

    import torch
    from transformers import GraniteConfig, GraniteForCausalLM

    config = GraniteConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
    )
    with torch.device("meta"):
        model = GraniteForCausalLM(config)

    module_names = [name for name, _ in model.named_modules()]

    for layer_map in DEFAULT_SMOOTHQUANT_MAPPINGS:
        smooth_pat = layer_map.smooth_layers.removeprefix("re:")
        smooth_re = re.compile(smooth_pat)
        smooth_hits = [n for n in module_names if smooth_re.search(n)]
        assert smooth_hits, (
            f"GraniteForCausalLM: smooth pattern {smooth_pat!r} matched no "
            f"modules; sample names: {module_names[:20]}"
        )
        for balance_pat_raw in layer_map.balance_layers:
            balance_pat = balance_pat_raw.removeprefix("re:")
            balance_re = re.compile(balance_pat)
            balance_hits = [n for n in module_names if balance_re.search(n)]
            assert balance_hits, (
                f"GraniteForCausalLM: balance pattern {balance_pat!r} matched "
                f"no modules; sample names: {module_names[:20]}"
            )
