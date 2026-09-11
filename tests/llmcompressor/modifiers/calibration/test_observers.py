import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import Observer


@pytest.mark.parametrize(
    "shape,group_size,actorder",
    [
        ((1, 1), None, False),
        ((1, 1), 1, False),
        ((1, 1), 1, "weight"),
        ((64, 64), None, False),
        ((64, 64), 32, False),
        ((64, 64), 32, "weight"),
        ((896, 4096), None, False),
        ((896, 4096), 7, False),
        ((896, 4096), 7, "weight"),
        ((512, 64), None, False),
        ((512, 64), 128, False),
        ((512, 64), 128, "weight"),
    ],
)
def test_observers_update(shape, group_size, actorder):
    module = torch.nn.Linear(*shape)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            group_size=group_size,
            actorder=actorder,
            observer="memoryless_minmax",
        ),
        input_activations=QuantizationArgs(observer="minmax"),
        output_activations=QuantizationArgs(observer="minmax"),
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
        name, base_name="input", args=QuantizationArgs(strategy="tensor"), **kwargs
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


def test_observe_skips_container_modules():
    """Container modules are modules, not iterables of modules to recurse into.

    `torch.nn.utils.parametrize` stores parametrizations in a `ModuleDict`, which is
    both a `Module` and an `Iterable`. Iterating it yields its keys, which are
    strings, so recursing into it never terminates. Every wav2vec2-family model
    weight-norms its positional convolution and so contains one.
    """
    model = torch.nn.Sequential(
        torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4, 4, 1)),
        torch.nn.Linear(4, 4),
    )
    linear = model[1]
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    initialize_module_for_quantization(
        linear, QuantizationScheme(targets=[], weights=args)
    )
    initialize_observer(linear, "weight")

    # pipelines pass every submodule of a subgraph, containers included
    modules = list(model.modules())
    assert any(isinstance(module, torch.nn.ModuleDict) for module in modules)

    observe(modules, "weight")
    update_qparams(modules, "weight")

    assert torch.isfinite(linear.weight_scale).all()
    assert linear.weight_scale > 0
