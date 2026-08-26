import pytest
import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.observers import Observer


def _args(**kwargs):
    values = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
        "zp_dtype": torch.int8,
    }
    values.update(kwargs)
    return QuantizationArgs(**values)


@pytest.mark.parametrize(
    "dynamic,expected_observer",
    [
        (False, "memoryless_minmax"),
        (DynamicType.LOCAL, "minmax"),
    ],
)
def test_initialize_observer_resolves_default(dynamic, expected_observer):
    module = torch.nn.Linear(4, 4)
    quant_args = {"dynamic": dynamic}
    if dynamic == DynamicType.LOCAL:
        quant_args.update(strategy="tensor_group", group_size=4)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        input_activations=_args(**quant_args),
    )

    initialize_observer(module, "input")

    assert module.quantization_scheme.input_activations.observer == expected_observer
    assert module.input_observer is not None


def test_initialize_observer_ignores_observer_for_dynamic_quantization():
    module = torch.nn.Linear(4, 4)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        input_activations=_args(
            strategy="token", dynamic=True, observer="static_minmax"
        ),
    )

    with pytest.warns(UserWarning, match="No observer is used"):
        initialize_observer(module, "input")

    assert module.quantization_scheme.input_activations.observer is None
    assert not hasattr(module, "input_observer")


def test_resolved_config_materializes_observer_policy():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=_args(),
        input_activations=_args(
            strategy="token", dynamic=True, observer="static_minmax"
        ),
    )

    with pytest.warns(UserWarning, match="No observer is used"):
        modifier = QuantizationModifier(config_groups={"group_0": scheme})

    resolved_scheme = modifier.resolved_config.config_groups["group_0"]
    assert resolved_scheme.weights.observer == "memoryless_minmax"
    assert resolved_scheme.input_activations.observer is None


@pytest.mark.parametrize(
    "shape,group_size,actorder",
    [
        ((1, 1), None, False),
        ((1, 1), 1, False),
        ((1, 1), 1, True),
        ((64, 64), None, False),
        ((64, 64), 32, False),
        ((64, 64), 32, True),
        ((896, 4096), None, False),
        ((896, 4096), 7, False),
        ((896, 4096), 7, True),
        ((512, 64), None, False),
        ((512, 64), 128, False),
        ((512, 64), 128, True),
    ],
)
def test_observers_update(shape, group_size, actorder):
    module = torch.nn.Linear(*shape)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=_args(
            strategy="group" if group_size is not None else "tensor",
            group_size=group_size,
            actorder=actorder,
        ),
        input_activations=_args(),
        output_activations=_args(),
    )

    input = torch.empty(module.in_features, dtype=module.weight.dtype)
    output = torch.empty(module.out_features, dtype=module.weight.dtype)

    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    initialize_observer(module, "input")
    initialize_observer(module, "output")

    for location, value in (
        ("weight", module.weight),
        ("input", input),
        ("output", output),
    ):
        observer = getattr(module, f"{location}_observer")
        qparams = observer(value).get_qparams()
        updated_scale = qparams["scale"]
        updated_zero_point = qparams["zero_point"]

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@pytest.mark.parametrize(
    "name,kwargs,observed,exp_min_vals,exp_max_vals",
    (
        (
            "memoryless_minmax",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-1.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "static_minmax",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-3.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "minmax",  # moving average
            {"averaging_constant": 0.1},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-0.3], [-0.37]]),
            torch.tensor([[0.0], [0.1], [0.39]]),
        ),
        (
            "memoryless_mse",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-1.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "mse",  # moving average
            {"averaging_constant": 0.1},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-0.3], [-0.37]]),
            torch.tensor([[0.0], [0.1], [0.39]]),
        ),
    ),
)
def test_observer_min_max_vals(name, kwargs, observed, exp_min_vals, exp_max_vals):
    observer = Observer.load_from_registry(
        name, base_name="input", args=_args(), **kwargs
    )

    min_vals, max_vals = [], []
    for _observed in observed:
        observer(_observed)
        _min_vals = observer.min_vals
        _max_vals = observer.max_vals

        min_vals.append(_min_vals)
        max_vals.append(_max_vals)

    min_vals = torch.stack(min_vals)
    max_vals = torch.stack(max_vals)
    assert torch.allclose(min_vals, exp_min_vals)
    assert torch.allclose(max_vals, exp_max_vals)
