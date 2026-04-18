"""
Unit tests for AWQModifier._assert_smoothing_health.

The health check is a pure function of the populated _error_metrics list,
so we drive it directly without a calibration loop.
"""

import pytest
from loguru import logger

from llmcompressor.modifiers.awq import AWQModifier


def _populate(awq: AWQModifier, metrics: list[dict]) -> None:
    """Pretend we just finished grid search."""
    awq._error_metrics.clear()
    awq._error_metrics.extend(metrics)


class _LoguruCapture:
    """Capture loguru records emitted while in the with-block."""

    def __init__(self, level: str = "WARNING"):
        self.level = level
        self.messages: list[str] = []
        self._handler_id: int | None = None

    def __enter__(self):
        def _sink(message):
            self.messages.append(message.record["message"])

        self._handler_id = logger.add(_sink, level=self.level, format="{message}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handler_id is not None:
            logger.remove(self._handler_id)


@pytest.mark.unit
def test_health_check_passes_when_all_layers_improve():
    """No warnings, no raise, when every layer's best_error < initial_error."""
    awq = AWQModifier(scheme="W4A16_ASYM", smoothing_health_max_error=1.0)
    _populate(
        awq,
        [
            {
                "layer_name": "layers.0.input_layernorm",
                "parent_name": "layers.0",
                "initial_error": 1.0,
                "best_error": 0.1,
                "reduction": 0.1,
            },
            {
                "layer_name": "layers.1.input_layernorm",
                "parent_name": "layers.1",
                "initial_error": 1.0,
                "best_error": 0.2,
                "reduction": 0.2,
            },
        ],
    )
    awq._assert_smoothing_health()  # should not raise


@pytest.mark.unit
def test_health_check_warns_on_increased_loss():
    """A layer where smoothing made things worse triggers a warning, not a raise."""
    awq = AWQModifier(scheme="W4A16_ASYM")  # smoothing_health_max_error=None
    _populate(
        awq,
        [
            {
                "layer_name": "layers.3.input_layernorm",
                "parent_name": "layers.3",
                "initial_error": 0.1,
                "best_error": 0.5,  # smoothing INCREASED loss
                "reduction": 5.0,
            },
        ],
    )
    with _LoguruCapture() as cap:
        awq._assert_smoothing_health()
    assert any("INCREASED loss" in m for m in cap.messages), cap.messages


@pytest.mark.unit
def test_health_check_raises_when_threshold_exceeded():
    """smoothing_health_max_error fires a RuntimeError on any over-threshold layer."""
    awq = AWQModifier(scheme="W4A16_ASYM", smoothing_health_max_error=1.0)
    _populate(
        awq,
        [
            {
                "layer_name": "layers.5.input_layernorm",
                "parent_name": "layers.5",
                "initial_error": 0.1,
                "best_error": 5.0,  # exceeds the bound
                "reduction": 50.0,
            },
        ],
    )
    with pytest.raises(RuntimeError, match="smoothing_health_max_error"):
        awq._assert_smoothing_health()


@pytest.mark.unit
def test_health_check_does_not_raise_when_threshold_unset():
    """Default smoothing_health_max_error=None is warn-only."""
    awq = AWQModifier(scheme="W4A16_ASYM")
    assert awq.smoothing_health_max_error is None
    _populate(
        awq,
        [
            {
                "layer_name": "layers.5.input_layernorm",
                "parent_name": "layers.5",
                "initial_error": 0.1,
                "best_error": 5.0,
                "reduction": 50.0,
            },
        ],
    )
    awq._assert_smoothing_health()  # should warn but not raise


@pytest.mark.unit
def test_health_check_handles_empty_metrics():
    """An AWQ run with no resolved mappings shouldn't crash the assertion."""
    awq = AWQModifier(scheme="W4A16_ASYM", smoothing_health_max_error=1.0)
    _populate(awq, [])
    awq._assert_smoothing_health()  # no-op


@pytest.mark.unit
def test_health_check_skips_layers_that_fell_back_to_identity():
    """
    When a layer was forced to identity scales by the max_scale_ratio guard
    its best_error == initial_error by construction (no smoothing applied).
    That is *not* a regression -- the fallback was already announced in
    _compute_best_scale -- so the health check must not double-warn nor
    trip the threshold gate.
    """
    awq = AWQModifier(scheme="W4A16_ASYM", smoothing_health_max_error=1.0)
    _populate(
        awq,
        [
            {
                "layer_name": "layers.7.mlp.up_proj",
                "parent_name": "layers.7.mlp",
                "initial_error": float("inf"),
                "best_error": float("inf"),
                "reduction": 1.0,
                "fell_back_to_identity": True,
            },
            {
                "layer_name": "layers.8.mlp.up_proj",
                "parent_name": "layers.8.mlp",
                "initial_error": 5.0,
                "best_error": 5.0,
                "reduction": 1.0,
                "fell_back_to_identity": True,
            },
        ],
    )
    with _LoguruCapture() as cap:
        awq._assert_smoothing_health()
    assert not any(
        "INCREASED loss" in m for m in cap.messages
    ), f"fallback layers must not be reported as INCREASED loss: {cap.messages}"


@pytest.mark.unit
def test_health_check_does_not_warn_on_no_change():
    """
    A layer where the grid search picked ratio=0 ends up with
    best_error == initial_error. Strictly speaking smoothing did not
    *increase* the loss, so no warning should fire.
    """
    awq = AWQModifier(scheme="W4A16_ASYM")
    _populate(
        awq,
        [
            {
                "layer_name": "layers.2.post_attention_layernorm",
                "parent_name": "layers.2",
                "initial_error": 6.976e-06,
                "best_error": 6.976e-06,
                "reduction": 1.0,
                "fell_back_to_identity": False,
            },
        ],
    )
    with _LoguruCapture() as cap:
        awq._assert_smoothing_health()
    assert not any(
        "INCREASED loss" in m for m in cap.messages
    ), f"best == initial is not a regression: {cap.messages}"
