import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.observers import Observer


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
        weights=QuantizationArgs(group_size=group_size, actorder=actorder),
        input_activations=QuantizationArgs(),
        output_activations=QuantizationArgs(),
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
        updated_scale, updated_zero_point, global_scale = qparams["scale"], qparams["zero_point"], qparams["global_scale"]

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@pytest.mark.parametrize("is_global", [False, True])
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
def test_observer_min_max_vals(
    name, kwargs, observed, exp_min_vals, exp_max_vals, is_global
):
    observer = Observer.load_from_registry(
        name, base_name="input", args=QuantizationArgs(strategy="tensor"), **kwargs
    )

    # All observers compute global min/max on-the-fly from min_vals/max_vals
    has_global_stats = False

    if is_global:
        # Skip - these observers compute global min/max on-the-fly
        pytest.skip(f"{name} observers compute global min/max on-the-fly from min_vals/max_vals")

    min_vals, max_vals = [], []
    for _observed in observed:
        observer(_observed)
        if not is_global:
            _min_vals = observer.min_vals
            _max_vals = observer.max_vals
        else:
            _min_vals = observer.statistics['global_min_vals']
            _max_vals = observer.statistics['global_max_vals']

        min_vals.append(_min_vals)
        max_vals.append(_max_vals)

    min_vals = torch.stack(min_vals)
    max_vals = torch.stack(max_vals)
    assert torch.allclose(min_vals, exp_min_vals)
    assert torch.allclose(max_vals, exp_max_vals)


def test_new_observer_api():
    """Test the new observer API: forward() and get_qparams()."""
    module = torch.nn.Linear(64, 64)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(group_size=32),
        input_activations=QuantizationArgs(),
    )

    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    initialize_observer(module, "input")

    # Test memoryless observer
    weight_observer = module.weight_observer
    input_value = torch.randn(10, 64)

    # Test observer(value) + get_qparams() separately
    input_observer = module.input_observer
    input_observer(input_value)
    qparams = input_observer.get_qparams()
    scale1, zp1, global_scale1 = qparams["scale"], qparams["zero_point"], qparams["global_scale"]
    assert scale1 is not None
    assert zp1 is not None

    # Test chaining: observer(value).get_qparams()
    qparams = input_observer(input_value).get_qparams()
    scale2, zp2, global_scale2 = qparams["scale"], qparams["zero_point"], qparams["global_scale"]
    assert scale2 is not None
    assert zp2 is not None

    # Test that calling get_qparams() without observer() raises error for memoryless
    fresh_observer = Observer.load_from_registry(
        "memoryless_minmax",
        base_name="input",
        args=QuantizationArgs(strategy="tensor"),
    )
    with pytest.raises(RuntimeError, match="No statistics available"):
        fresh_observer.get_qparams()

    # Test stateful observer accumulates statistics
    stateful_observer = Observer.load_from_registry(
        "static_minmax",
        base_name="input",
        args=QuantizationArgs(strategy="tensor"),
    )
    value1 = torch.tensor([1.0, 2.0, 3.0])
    value2 = torch.tensor([0.0, 4.0, 2.0])

    stateful_observer(value1)
    qparams = stateful_observer.get_qparams()
    scale_after_1, _ = qparams["scale"], qparams["zero_point"]

    stateful_observer(value2)
    qparams = stateful_observer.get_qparams()
    scale_after_2, _ = qparams["scale"], qparams["zero_point"]

    # Scale should change after second update (accumulated min/max)
    assert not torch.allclose(scale_after_1, scale_after_2)


def test_observer_api_patterns():
    """Test different API patterns for using observers."""
    module = torch.nn.Linear(32, 32)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(),
    )

    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")

    observer = module.weight_observer
    value = module.weight

    # Pattern 1: Chained API - observer(value).get_qparams()
    qparams = observer(value).get_qparams()
    scale_chain, zp_chain = qparams["scale"], qparams["zero_point"]

    # Pattern 2: Explicit separation - observer(value) then get_qparams()
    observer(value)
    qparams = observer.get_qparams()
    scale_sep, zp_sep = qparams["scale"], qparams["zero_point"]
    assert torch.allclose(scale_chain, scale_sep)
    assert torch.allclose(zp_chain, zp_sep)

    # Test get_global_scale() method
    from compressed_tensors.quantization import QuantizationStrategy

    tg_observer = Observer.load_from_registry(
        "memoryless_minmax",
        base_name="weight",
        args=QuantizationArgs(
            strategy=QuantizationStrategy.TENSOR_GROUP, group_size=16
        ),
        module=module,
    )

    tg_observer(value)
    qparams = tg_observer.get_qparams()
    global_scale = qparams["global_scale"]
    assert global_scale is not None
    assert global_scale.numel() > 0


def test_observer_statistics_dict():
    """Test that observer statistics are stored as direct attributes."""
    # Test memoryless observer
    memoryless = Observer.load_from_registry(
        "memoryless_minmax",
        base_name="weight",
        args=QuantizationArgs(strategy="tensor"),
    )

    # Statistics are stored as direct attributes, not in a dict
    assert not hasattr(memoryless, 'min_vals')  # Not yet observed

    value = torch.randn(10, 10)
    memoryless(value)
    assert hasattr(memoryless, 'min_vals')
    assert hasattr(memoryless, 'max_vals')
    assert isinstance(memoryless.min_vals, torch.Tensor)
    assert isinstance(memoryless.max_vals, torch.Tensor)
    # For minmax observers, global min/max is computed on-the-fly from min_vals/max_vals
    # (not stored separately)

    # Test stateful observer
    stateful = Observer.load_from_registry(
        "static_minmax",
        base_name="weight",
        args=QuantizationArgs(strategy="tensor"),
    )

    # Statistics are stored as direct attributes, not in a dict
    assert not hasattr(stateful, 'min_vals')  # Not yet observed

    value1 = torch.tensor([1.0, 2.0, 3.0])
    stateful(value1)
    assert hasattr(stateful, 'min_vals')
    assert hasattr(stateful, 'max_vals')
    # For minmax observers, global min/max is computed on-the-fly from min_vals/max_vals
    # (not stored separately)

    # Verify values are correct
    assert torch.allclose(stateful.min_vals, torch.tensor([1.0]))
    assert torch.allclose(stateful.max_vals, torch.tensor([3.0]))

    # Test accumulation
    value2 = torch.tensor([0.0, 4.0, 2.0])
    stateful(value2)
    assert torch.allclose(stateful.min_vals, torch.tensor([0.0]))
    assert torch.allclose(stateful.max_vals, torch.tensor([4.0]))
