import pytest
import torch
from loguru import logger

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier


def _config_groups():
    return {
        "group_0": {
            "targets": ["Linear"],
            "weights": {},
            "input_activations": {},
            "output_activations": {},
        }
    }


def test_observer_defaults_are_based_on_scheme_location():
    modifier = QuantizationModifier(config_groups=_config_groups())

    scheme = modifier.resolve_quantization_config().config_groups["group_0"]
    assert scheme.weights.observer == "memoryless_minmax"
    assert scheme.input_activations.observer == "minmax"
    assert scheme.output_activations.observer == "minmax"


def test_weight_observer_defaults_to_imatrix_with_calibration_data():
    modifier = QuantizationModifier(config_groups=_config_groups())

    scheme = modifier.resolve_quantization_config(
        has_calibration_data=True
    ).config_groups["group_0"]
    assert scheme.weights.observer == "imatrix_mse"
    assert scheme.input_activations.observer == "minmax"
    assert scheme.output_activations.observer == "minmax"


def test_initialize_quantization_uses_state_calibration_data():
    def make_modifier_and_model():
        model = torch.nn.Sequential(torch.nn.Linear(128, 128))
        modifier = QuantizationModifier(targets="Linear", scheme="W4A16")
        return modifier, model

    # no calibration data -> memoryless minmax
    modifier, model = make_modifier_and_model()
    modifier.initialize_quantization(State(model=model))
    assert model[0].quantization_scheme.weights.observer == "memoryless_minmax"

    # with calibration data -> imatrix
    modifier, model = make_modifier_and_model()
    state = State(model=model)
    state.data.calib = [torch.randn(2, 128)]  # single calibration sample
    modifier.initialize_quantization(state)
    assert model[0].quantization_scheme.weights.observer == "imatrix_mse"


def test_resolution_does_not_mutate_modifier_config_groups():
    modifier = QuantizationModifier(config_groups=_config_groups())

    modifier.resolve_quantization_config(has_calibration_data=True)
    modifier.resolve_quantization_config()

    # fields specified on the modifier are left untouched so that resolution
    # can be repeated with different `has_calibration_data` values
    scheme = modifier.config_groups["group_0"]
    assert scheme.weights.observer is None
    assert scheme.input_activations.observer is None
    assert scheme.output_activations.observer is None


def test_scheme_observer_is_preserved():
    modifier = QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {"observer": "memoryless_mse"},
                "input_activations": {"observer": "static_minmax"},
            }
        }
    )

    scheme = modifier.resolve_quantization_config(
        has_calibration_data=True
    ).config_groups["group_0"]
    assert scheme.weights.observer == "memoryless_mse"
    assert scheme.input_activations.observer == "static_minmax"


def test_modifier_observer_overrides_scheme():
    modifier = QuantizationModifier(
        input_observer="mse",
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {"observer": "static_minmax"},
            }
        },
    )

    scheme = modifier.resolve_quantization_config().config_groups["group_0"]
    assert scheme.input_activations.observer == "mse"


def test_static_fp8_input_uses_minmax():
    modifier = QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {"type": "float", "num_bits": 8},
            }
        }
    )
    scheme = modifier.resolve_quantization_config().config_groups["group_0"]

    assert scheme.input_activations.dynamic is False
    assert scheme.input_activations.observer == "minmax"


def test_serialized_null_observer_gets_default():
    modifier = QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {"observer": None},
                "input_activations": {"observer": None},
                "output_activations": {"observer": None},
            }
        }
    )

    scheme = modifier.resolve_quantization_config().config_groups["group_0"]
    assert scheme.weights.observer == "memoryless_minmax"
    assert scheme.input_activations.observer == "minmax"
    assert scheme.output_activations.observer == "minmax"

    scheme = modifier.resolve_quantization_config(
        has_calibration_data=True
    ).config_groups["group_0"]
    assert scheme.weights.observer == "imatrix_mse"
    assert scheme.input_activations.observer == "minmax"
    assert scheme.output_activations.observer == "minmax"


@pytest.mark.parametrize(
    "has_calibration_data", [False, True], ids=["no_calib", "calib"]
)
def test_info_log_when_observer_default_is_overridden(has_calibration_data):
    modifier = QuantizationModifier(
        weight_observer="minmax",
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {},
                "input_activations": {"observer": "mse"},
            }
        },
    )

    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="INFO")
    try:
        modifier.resolve_quantization_config(has_calibration_data=has_calibration_data)
    finally:
        logger.remove(handler_id)

    default_weight_obs = "imatrix_mse" if has_calibration_data else "memoryless_minmax"
    assert any(
        f"Overriding default weight observer '{default_weight_obs}' with 'minmax'"
        in log
        for log in logs
    )
    assert any(
        "Overriding default input activation observer 'minmax' with 'mse'" in log
        for log in logs
    )


def test_no_info_log_when_observer_defaults_are_applied():
    modifier = QuantizationModifier(config_groups=_config_groups())

    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="INFO")
    try:
        modifier.resolve_quantization_config(has_calibration_data=True)
        modifier.resolve_quantization_config()
    finally:
        logger.remove(handler_id)

    assert not any("Overriding default" in log for log in logs)
