import warnings

import pytest
import torch
import torch.nn as nn

from llmcompressor.utils.pytorch import (
    build_parameterized_layers,
    expand_special_targets,
)


@pytest.fixture
def simple_model():
    """Simple model for testing parameterized layers."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    return model


@pytest.mark.unit
def test_expand_special_targets():
    """Test expand_special_targets function with special constants."""
    # Test __ALL_PRUNABLE__ expansion
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = expand_special_targets("__ALL_PRUNABLE__")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Linear" in result
        assert "Conv1d" in result
        assert "Conv2d" in result
        assert "Conv3d" in result

    # Test __ALL_QUANTIZABLE__ expansion
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = expand_special_targets("__ALL_QUANTIZABLE__")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Linear" in result
        assert "Conv2d" in result
        assert "Conv3d" in result
        assert "Conv1d" not in result  # Conv1d is not quantizable

    # Test __ALL__ raises error
    with pytest.raises(ValueError, match="no longer supported"):
        expand_special_targets("__ALL__")

    # Test normal targets pass through unchanged
    result = expand_special_targets(["Linear", "re:.*attn.*"])
    assert result == ["Linear", "re:.*attn.*"]

    # Test mixed targets
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = expand_special_targets(["Linear", "__ALL_PRUNABLE__", "Conv2d"])
        assert len(w) == 1
        assert "Linear" in result
        assert "Conv1d" in result
        assert "Conv2d" in result
        assert "Conv3d" in result


@pytest.mark.unit
def test_build_parameterized_layers(simple_model):
    """Test build_parameterized_layers function."""
    # Initialize weights so we can test
    for module in simple_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.ones_(module.weight)

    # Test with Linear target
    result = build_parameterized_layers(simple_model, ["Linear"])

    # Should have 2 Linear layers
    assert len(result) == 2
    assert "0" in result
    assert "2" in result

    # Check ModelParameterizedLayer structure
    layer_0 = result["0"]
    assert layer_0.layer_name == "0"
    assert isinstance(layer_0.layer, nn.Linear)
    assert layer_0.param_name == "0.weight"
    assert isinstance(layer_0.param, torch.nn.Parameter)

    # Test with specific weight parameter
    result = build_parameterized_layers(simple_model, ["Linear"], param_name="weight")
    assert len(result) == 2

    # Test with non-existent parameter name (should return empty)
    result = build_parameterized_layers(
        simple_model, ["Linear"], param_name="non_existent"
    )
    assert len(result) == 0
