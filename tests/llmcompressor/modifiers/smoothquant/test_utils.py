from unittest.mock import patch

import pytest

from llmcompressor.modifiers.smoothquant.utils import (
    get_layer_mappings_from_architecture,
    handle_mapping_resolution_errors,
)

smoothquant_utils = "llmcompressor.modifiers.smoothquant.utils"


@pytest.mark.unit
def test_handle_mapping_resolution_errors():
    README_LOCATION = (
        "https://github.com/vllm-project/llm-compressor/tree/main/"
        "src/llmcompressor/modifiers/smoothquant"
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
