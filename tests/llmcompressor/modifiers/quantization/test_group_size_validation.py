"""Tests for early group-size divisibility validation."""

import pytest
import torch

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.group_size_validation import (
    get_layers_indivisible_by_group_size,
)


def _make_tiny_model(columns: int, divisible_columns: int | None = None):
    """Model with one Linear with columns, optionally another with divisible_columns."""
    linears = {"indiv": torch.nn.Linear(64, columns)}
    if divisible_columns is not None:
        linears["div"] = torch.nn.Linear(64, divisible_columns)
    return torch.nn.ModuleDict(linears)


def test_get_layers_indivisible_by_group_size_empty():
    """When all layers are divisible, helper returns empty list."""
    from compressed_tensors.quantization import (
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStatus,
        apply_quantization_config,
    )
    from compressed_tensors.quantization.quant_args import QuantizationArgs

    model = _make_tiny_model(128)  # 128 % 128 == 0
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(strategy="group", group_size=128),
    )
    config = QuantizationConfig(
        config_groups={"g": scheme},
        kv_cache_scheme=None,
        quantization_status=QuantizationStatus.INITIALIZED,
        ignore=[],
    )
    apply_quantization_config(model, config)
    out = get_layers_indivisible_by_group_size(model, {"Linear"}, [])
    assert out == []


def test_get_layers_indivisible_by_group_size_finds_layer():
    """Helper returns (fqn, columns, group_size) for indivisible layers."""
    from compressed_tensors.quantization import (
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStatus,
        apply_quantization_config,
    )
    from compressed_tensors.quantization.quant_args import QuantizationArgs

    model = _make_tiny_model(100)  # 100 % 128 != 0
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(strategy="group", group_size=128),
    )
    config = QuantizationConfig(
        config_groups={"g": scheme},
        kv_cache_scheme=None,
        quantization_status=QuantizationStatus.INITIALIZED,
        ignore=[],
    )
    apply_quantization_config(model, config)
    out = get_layers_indivisible_by_group_size(model, {"Linear"}, [])
    assert len(out) == 1
    fqn, cols, gs = out[0]
    assert "indiv" in fqn
    assert cols == 100
    assert gs == 128


def test_initialize_quantization_raises_early_for_indivisible():
    """Modifier raises at on_initialize with clear message and layer names."""
    model = _make_tiny_model(100)
    state = State()
    state.update(model=model, device="cpu")
    modifier = QuantizationModifier(scheme="W4A16", targets=["Linear"])

    with torch.no_grad():
        with pytest.raises(ValueError) as exc_info:
            modifier.on_initialize(state)

    msg = str(exc_info.value)
    assert "columns" in msg.lower() and "group_size" in msg.lower()
    assert "ignore" in msg.lower()
    assert "indiv" in msg
    assert "100" in msg and "128" in msg


def test_initialize_quantization_succeeds_when_indivisible_ignored():
    """When indivisible layer is in ignore list, on_initialize does not raise."""
    model = _make_tiny_model(100)
    state = State()
    state.update(model=model, device="cpu")
    # Match the actual FQN: our model has "indiv" and "div"; the Linear is under "indiv"
    modifier = QuantizationModifier(
        scheme="W4A16", targets=["Linear"], ignore=["indiv"]
    )

    with torch.no_grad():
        modifier.on_initialize(state)

    # No exception; quantization was applied only to layers that are divisible (none
    # in this model since we ignored the only Linear). So config is applied, validation
    # sees no targeted indivisible layers.
    assert True


def test_initialize_quantization_succeeds_when_all_divisible():
    """When all targeted layers have columns % group_size == 0, no error."""
    model = _make_tiny_model(256)  # 256 % 128 == 0
    state = State()
    state.update(model=model, device="cpu")
    modifier = QuantizationModifier(scheme="W4A16", targets=["Linear"])

    with torch.no_grad():
        modifier.on_initialize(state)

    assert True
