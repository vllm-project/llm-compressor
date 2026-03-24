"""
Tests for AWQModifier recipe validation and temporary quant scheme management.
"""

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from torch.nn import Linear

from llmcompressor.core import State, active_session
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier
from llmcompressor.recipe import Recipe


def _make_simple_model():
    """Create a simple model with Linear layers for testing."""
    return torch.nn.Sequential(
        torch.nn.LayerNorm(8),
        Linear(8, 8),
        Linear(8, 8),
    )


def _setup_session_with_modifiers(modifiers):
    """
    Register modifiers on the active session's lifecycle recipe so that
    _validate_recipe (which calls active_session().lifecycle.recipe.modifiers)
    can find them. Returns a State with the model attached.
    """
    session = active_session()
    session.lifecycle.recipe = Recipe.from_modifiers(modifiers)
    state = State(model=_make_simple_model())
    return state


# ------------------------------------------------------------------ #
#  Validation tests                                                    #
# ------------------------------------------------------------------ #


@pytest.mark.unit
def test_validate_warns_no_downstream_quantizer():
    """AWQ without a downstream quantizer should log a warning."""
    from loguru import logger

    awq = AWQModifier(scheme="W4A16")
    state = _setup_session_with_modifiers([awq])

    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        awq._validate_recipe(state)
    finally:
        logger.remove(handler_id)

    assert any("without a downstream quantizer" in m for m in messages)


@pytest.mark.unit
def test_validate_warns_reversed_order():
    """Quantizer before AWQ should warn about reversed ordering."""
    from loguru import logger

    quant = QuantizationModifier(scheme="W4A16")
    awq = AWQModifier(scheme="W4A16")
    state = _setup_session_with_modifiers([quant, awq])

    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        awq._validate_recipe(state)
    finally:
        logger.remove(handler_id)

    assert any("before AWQModifier" in m for m in messages)


@pytest.mark.unit
def test_validate_passes_correct_order():
    """AWQ followed by QuantizationModifier should pass validation."""
    awq = AWQModifier(scheme="W4A16")
    quant = QuantizationModifier(scheme="W4A16")
    state = _setup_session_with_modifiers([awq, quant])

    # Should not raise
    awq._validate_recipe(state)


@pytest.mark.unit
def test_validate_mismatched_num_bits():
    """Mismatched num_bits between AWQ and downstream quantizer should error."""
    awq = AWQModifier(scheme="W4A16")
    quant = QuantizationModifier(scheme="W8A16")
    state = _setup_session_with_modifiers([awq, quant])

    with pytest.raises(ValueError, match="num_bits"):
        awq._validate_recipe(state)


@pytest.mark.unit
def test_validate_mismatched_symmetry():
    """Mismatched symmetry should error."""
    awq = AWQModifier(scheme="W4A16_ASYM")

    # Create a symmetric quantizer via config_groups
    quant = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    symmetric=True,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                ),
            )
        }
    )
    state = _setup_session_with_modifiers([awq, quant])

    with pytest.raises(ValueError, match="symmetric"):
        awq._validate_recipe(state)


@pytest.mark.unit
def test_validate_warns_mismatched_ignore():
    """Mismatched ignore lists should warn."""
    from loguru import logger

    awq = AWQModifier(scheme="W4A16", ignore=["lm_head"])
    quant = QuantizationModifier(scheme="W4A16", ignore=["lm_head", "extra_layer"])
    state = _setup_session_with_modifiers([awq, quant])

    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        awq._validate_recipe(state)
    finally:
        logger.remove(handler_id)

    assert any("ignore list" in m for m in messages)


# ------------------------------------------------------------------ #
#  Temporary quant scheme context manager tests                        #
# ------------------------------------------------------------------ #


@pytest.mark.unit
def test_temporary_quant_schemes_applied_and_stripped():
    """
    Verify that _temporary_quant_schemes applies schemes inside the context
    and fully strips them on exit.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")

    # Before: no quant schemes
    for module in model.modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "quantization_scheme")

    with awq._temporary_quant_schemes(model):
        # Inside: quant schemes should be present
        for module in model.modules():
            if isinstance(module, Linear):
                assert hasattr(module, "quantization_scheme")

    # After: quant schemes should be stripped
    for module in model.modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "quantization_scheme")


@pytest.mark.unit
def test_temporary_quant_schemes_preserves_prior_state():
    """
    If a downstream modifier already applied quant schemes, the context
    manager should restore them exactly on exit.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")

    # Simulate a prior modifier having set a scheme on one layer
    linear = list(m for m in model.modules() if isinstance(m, Linear))[0]
    prior_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8),
    )
    linear.quantization_scheme = prior_scheme

    with awq._temporary_quant_schemes(model):
        # AWQ's temp scheme is present (may differ from prior)
        assert hasattr(linear, "quantization_scheme")

    # After exit: prior scheme should be restored
    assert hasattr(linear, "quantization_scheme")
    assert linear.quantization_scheme is prior_scheme


@pytest.mark.unit
def test_temporary_quant_schemes_strips_on_exception():
    """
    If an exception occurs inside the context, schemes should still be stripped.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")

    with pytest.raises(RuntimeError):
        with awq._temporary_quant_schemes(model):
            # Verify schemes are present
            for module in model.modules():
                if isinstance(module, Linear):
                    assert hasattr(module, "quantization_scheme")
            raise RuntimeError("simulated failure")

    # After exception: schemes should be stripped
    for module in model.modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "quantization_scheme")


@pytest.mark.unit
def test_temporary_quant_schemes_with_observers_true():
    """
    When with_observers=True, weight observers should be initialized on
    modules that have a weight quantization scheme.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")

    with awq._temporary_quant_schemes(model, with_observers=True):
        for module in model.modules():
            if isinstance(module, Linear):
                assert hasattr(module, "quantization_scheme")
                if module.quantization_scheme.weights is not None:
                    assert hasattr(module, "weight_observer"), (
                        "weight_observer should be initialized when with_observers=True"
                    )

    # After exit: observers should be stripped along with schemes
    for module in model.modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "weight_observer")


@pytest.mark.unit
def test_temporary_quant_schemes_with_observers_false():
    """
    When with_observers=False (default), weight observers should NOT be
    initialized. Only scheme metadata should be present.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")

    with awq._temporary_quant_schemes(model, with_observers=False):
        for module in model.modules():
            if isinstance(module, Linear):
                assert hasattr(module, "quantization_scheme")
                # No observer should be created
                assert not hasattr(module, "weight_observer"), (
                    "weight_observer should NOT be initialized when "
                    "with_observers=False"
                )


@pytest.mark.unit
def test_on_initialize_leaves_no_quant_schemes():
    """
    After on_initialize returns, no temporary quant schemes should remain
    on the model. This verifies on_initialize uses the context manager
    properly and does not leak state.
    """
    model = _make_simple_model()
    awq = AWQModifier(scheme="W4A16")
    quant = QuantizationModifier(scheme="W4A16")

    session = active_session()
    session.lifecycle.recipe = Recipe.from_modifiers([awq, quant])
    state = State(model=model)

    awq.on_initialize(state)

    for module in model.modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "quantization_scheme"), (
                "on_initialize should not leave quantization_scheme on modules"
            )
            assert not hasattr(module, "quantization_status"), (
                "on_initialize should not leave quantization_status on modules"
            )
            assert not hasattr(module, "weight_observer"), (
                "on_initialize should not leave weight_observer on modules"
            )


@pytest.mark.unit
def test_on_initialize_preserves_downstream_modifier_schemes():
    """
    If a downstream modifier has already applied quant schemes before AWQ
    initializes, on_initialize must not destroy them.
    """
    model = _make_simple_model()
    linear = list(m for m in model.modules() if isinstance(m, Linear))[0]

    # Simulate a downstream modifier having already set a scheme
    prior_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4),
    )
    linear.quantization_scheme = prior_scheme

    awq = AWQModifier(scheme="W4A16")
    quant = QuantizationModifier(scheme="W4A16")

    session = active_session()
    session.lifecycle.recipe = Recipe.from_modifiers([awq, quant])
    state = State(model=model)

    awq.on_initialize(state)

    # Prior scheme must be restored
    assert hasattr(linear, "quantization_scheme")
    assert linear.quantization_scheme is prior_scheme


# ------------------------------------------------------------------ #
#  AWQModifier no longer inherits QuantizationMixin                    #
# ------------------------------------------------------------------ #


@pytest.mark.unit
def test_awq_does_not_inherit_quantization_mixin():
    """Confirm AWQModifier no longer inherits from QuantizationMixin."""
    from llmcompressor.modifiers.quantization.quantization import QuantizationMixin

    assert not issubclass(AWQModifier, QuantizationMixin)


@pytest.mark.unit
def test_awq_is_smoothing_only():
    """
    AWQModifier.on_end should NOT call any quantization finalization methods.
    Verify the modifier's on_end is clean — just asserts activations consumed,
    sets ended_ flag, and removes hooks.
    """
    awq = AWQModifier(scheme="W4A16")
    # Simulate that smoothing has already run and activations consumed
    awq._smooth_activation_means = {}
    awq.started_ = True

    state = State(model=_make_simple_model())

    # on_end should not raise or try to call quantization methods
    awq.on_end(state, None)
    assert awq.ended_ is True
