import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.utils import generate_gparam
from torch.nn import Linear, Module

from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    update_qparams,
)
from llmcompressor.observers.helpers import fuse_weight_observers

TENSOR_GROUP_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type="float",
        strategy="tensor_group",
        group_size=16,
    ),
)

NON_TENSOR_GROUP_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type="int",
        strategy="channel",
    ),
)


class MockAttentionModule(Module):
    """Mock attention module with q_proj, k_proj, v_proj."""

    def __init__(self, scheme: QuantizationScheme):
        super().__init__()
        # Use different weight magnitudes so global_scales would differ
        self.q_proj = Linear(64, 64, bias=False)
        self.k_proj = Linear(64, 64, bias=False)
        self.v_proj = Linear(64, 64, bias=False)

        torch.manual_seed(42)
        self.q_proj.weight.data = torch.randn(64, 64) * 1.0
        self.k_proj.weight.data = torch.randn(64, 64) * 2.0  # larger magnitude
        self.v_proj.weight.data = torch.randn(64, 64) * 0.5  # smaller magnitude

        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            initialize_module_for_quantization(proj, scheme)


class MockMLPModule(Module):
    """Mock MLP module with gate_proj and up_proj."""

    def __init__(self, scheme: QuantizationScheme):
        super().__init__()
        self.gate_proj = Linear(64, 128, bias=False)
        self.up_proj = Linear(64, 128, bias=False)

        torch.manual_seed(99)
        self.gate_proj.weight.data = torch.randn(128, 64) * 1.5
        self.up_proj.weight.data = torch.randn(128, 64) * 3.0

        for proj in [self.gate_proj, self.up_proj]:
            initialize_module_for_quantization(proj, scheme)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_class,layer_names",
    [
        (MockAttentionModule, ["q_proj", "k_proj", "v_proj"]),
        (MockMLPModule, ["gate_proj", "up_proj"]),
    ],
)
def test_fused_observers_produce_identical_global_scale(module_class, layer_names):
    """
    Test that fused observers compute the same global_scale for all
    observers in a fused group, and that it matches the expected value
    (computed from the combined absmax of all weights).
    """
    module = module_class(scheme=TENSOR_GROUP_SCHEME)

    # Initialize observers and link them
    for name in layer_names:
        layer = getattr(module, name)
        initialize_observer(layer, "weight")
    fuse_weight_observers(module)

    # Observe weights
    for name in layer_names:
        layer = getattr(module, name)
        layer.weight_observer(layer.weight)

    # Compute qparams (all fused partners now have statistics)
    for name in layer_names:
        layer = getattr(module, name)
        update_qparams(layer, base_name="weight")

    # All layers should have the same global_scale
    layers = [getattr(module, name) for name in layer_names]
    global_scales = [layer.weight_global_scale.data for layer in layers]
    for i in range(1, len(global_scales)):
        assert torch.allclose(global_scales[0], global_scales[i]), (
            f"Layer {layer_names[i]} global_scale {global_scales[i].item():.6f} "
            f"differs from {layer_names[0]} {global_scales[0].item():.6f}"
        )

    # Verify the fused global_scale matches the expected value from combined absmax
    all_absmax = max(layer.weight.data.abs().max().item() for layer in layers)
    expected_global_scale = generate_gparam(
        torch.tensor([-all_absmax]), torch.tensor([all_absmax])
    )
    assert torch.allclose(global_scales[0], expected_global_scale, rtol=1e-5), (
        f"Fused global_scale {global_scales[0].item():.6f} doesn't match "
        f"expected {expected_global_scale.item():.6f}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_class,layer_names",
    [
        (MockAttentionModule, ["q_proj", "k_proj", "v_proj"]),
        (MockMLPModule, ["gate_proj", "up_proj"]),
    ],
)
def test_unfused_observers_produce_different_global_scale(module_class, layer_names):
    """
    Test that without fusion, observers compute different global_scales
    (since each weight has different magnitude).
    """
    module = module_class(scheme=TENSOR_GROUP_SCHEME)

    # Initialize observers WITHOUT linking
    for name in layer_names:
        layer = getattr(module, name)
        initialize_observer(layer, "weight")

    # Observe weights and compute qparams
    for name in layer_names:
        layer = getattr(module, name)
        layer.weight_observer(layer.weight)
        update_qparams(layer, base_name="weight")

    # Layers should have different global_scales
    layers = [getattr(module, name) for name in layer_names]
    global_scales = [layer.weight_global_scale.data for layer in layers]
    # At least one pair should differ (weights have different magnitudes)
    all_same = all(torch.allclose(global_scales[0], gs) for gs in global_scales[1:])
    assert not all_same, "Expected different global_scales without fusion"


@pytest.mark.unit
def test_non_tensor_group_not_affected():
    """
    Test that fuse_weight_observers does not link observers for
    non-TENSOR_GROUP quantization strategies.
    """
    module = MockAttentionModule(scheme=NON_TENSOR_GROUP_SCHEME)

    for name in ["q_proj", "k_proj", "v_proj"]:
        layer = getattr(module, name)
        initialize_observer(layer, "weight")

    # This should be a no-op for non-TENSOR_GROUP
    fuse_weight_observers(module)

    # Verify observers have no fused partners
    for name in ["q_proj", "k_proj", "v_proj"]:
        layer = getattr(module, name)
        assert len(layer.weight_observer._fused_observers) == 0


@pytest.mark.unit
def test_fused_recompute_after_weight_change():
    """
    Test that after linking, recomputing qparams with a changed weight
    still produces a correct fused global_scale.
    """
    module = MockAttentionModule(scheme=TENSOR_GROUP_SCHEME)

    for name in ["q_proj", "k_proj", "v_proj"]:
        layer = getattr(module, name)
        initialize_observer(layer, "weight")
    fuse_weight_observers(module)

    # Initial observation
    for name in ["q_proj", "k_proj", "v_proj"]:
        layer = getattr(module, name)
        layer.weight_observer(layer.weight)
    for name in ["q_proj", "k_proj", "v_proj"]:
        layer = getattr(module, name)
        update_qparams(layer, base_name="weight")

    initial_gs = module.q_proj.weight_global_scale.data.clone()

    # Change q_proj weight to have much larger magnitude
    with torch.no_grad():
        module.q_proj.weight *= 10.0
    module.q_proj.weight_observer(module.q_proj.weight)
    update_qparams(module.q_proj, base_name="weight")

    # global_scale should have changed (larger weight = smaller global_scale)
    new_gs = module.q_proj.weight_global_scale.data
    assert not torch.allclose(
        initial_gs, new_gs
    ), "global_scale should change after weight magnitude change"

    # Recompute k_proj — should see q_proj's new statistics via fusion
    module.k_proj.weight_observer(module.k_proj.weight)
    update_qparams(module.k_proj, base_name="weight")
    assert torch.allclose(
        module.q_proj.weight_global_scale.data, module.k_proj.weight_global_scale.data
    ), "Fused observers should still produce same global_scale after recompute"
