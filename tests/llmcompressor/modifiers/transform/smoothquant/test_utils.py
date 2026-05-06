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
