from llmcompressor.modifiers.quantization import QuantizationModifier


def test_observer_defaults_are_based_on_scheme_location():
    modifier = QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {},
                "input_activations": {},
                "output_activations": {},
            }
        }
    )

    scheme = modifier.resolved_config.config_groups["group_0"]
    assert scheme.weights.observer == "memoryless_minmax"
    assert scheme.input_activations.observer == "minmax"
    assert scheme.output_activations.observer == "minmax"


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

    scheme = modifier.resolved_config.config_groups["group_0"]
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

    scheme = modifier.resolved_config.config_groups["group_0"]
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
    scheme = modifier.resolved_config.config_groups["group_0"]

    assert scheme.input_activations.dynamic is False
    assert scheme.input_activations.observer == "minmax"
