import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.nn import Linear, Module

from llmcompressor.modifiers.utils.helpers import (
    update_fused_layer_weight_global_scales,
)


class MockAttentionModule(Module):
    """Mock attention module with q_proj, k_proj, v_proj."""

    def __init__(self, tensor_group_quant: bool = True):
        super().__init__()
        self.q_proj = Linear(128, 128, bias=False)
        self.k_proj = Linear(128, 128, bias=False)
        self.v_proj = Linear(128, 128, bias=False)

        if tensor_group_quant:
            scheme = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="float",
                    strategy="tensor_group",
                    group_size=128,
                ),
            )
            for proj in [self.q_proj, self.k_proj, self.v_proj]:
                proj.quantization_scheme = scheme
                proj.weight_global_scale = torch.nn.Parameter(
                    torch.randn(1), requires_grad=False
                )
                # Add weight_scale for fusion to work
                num_groups = proj.weight.shape[1] // 128
                proj.weight_scale = torch.nn.Parameter(
                    torch.ones(proj.weight.shape[0], num_groups), requires_grad=False
                )


class MockMLPModule(Module):
    """Mock MLP module with gate_proj and up_proj."""

    def __init__(self, tensor_group_quant: bool = True):
        super().__init__()
        self.gate_proj = Linear(128, 256, bias=False)
        self.up_proj = Linear(128, 256, bias=False)

        if tensor_group_quant:
            scheme = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="float",
                    strategy="tensor_group",
                    group_size=128,
                ),
            )
            for proj in [self.gate_proj, self.up_proj]:
                proj.quantization_scheme = scheme
                proj.weight_global_scale = torch.nn.Parameter(
                    torch.randn(1), requires_grad=False
                )
                # Add weight_scale for fusion to work
                num_groups = proj.weight.shape[1] // 128
                proj.weight_scale = torch.nn.Parameter(
                    torch.ones(proj.weight.shape[0], num_groups), requires_grad=False
                )


class MockDeepSeekMLAModule(Module):
    """Mock DeepSeek multi-latent attention module."""

    def __init__(self, tensor_group_quant: bool = True):
        super().__init__()
        self.q_a_proj = Linear(128, 128, bias=False)
        self.kv_a_proj_with_mqa = Linear(128, 256, bias=False)

        if tensor_group_quant:
            scheme = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="float",
                    strategy="tensor_group",
                    group_size=128,
                ),
            )
            for proj in [self.q_a_proj, self.kv_a_proj_with_mqa]:
                proj.quantization_scheme = scheme
                proj.weight_global_scale = torch.nn.Parameter(
                    torch.randn(1), requires_grad=False
                )
                # Add weight_scale for fusion to work
                num_groups = proj.weight.shape[1] // 128
                proj.weight_scale = torch.nn.Parameter(
                    torch.ones(proj.weight.shape[0], num_groups), requires_grad=False
                )


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_class,layer_names",
    [
        (MockAttentionModule, ["q_proj", "k_proj", "v_proj"]),
        (MockMLPModule, ["gate_proj", "up_proj"]),
        (MockDeepSeekMLAModule, ["q_a_proj", "kv_a_proj_with_mqa"]),
    ],
)
@pytest.mark.parametrize("tensor_group_quant", [False, True])
def test_update_fused_layer_weight_global_scales_fuses_scales(
    module_class, layer_names, tensor_group_quant
):
    """
    Test that update_fused_layer_weight_global_scales correctly fuses
    weight_global_scale across related layers.
    """
    # different global scales are set for each layer in constructor
    module = module_class(tensor_group_quant=tensor_group_quant)

    # Run the function, should still run successfully if tensor_group_quant is False
    update_fused_layer_weight_global_scales(module)

    if tensor_group_quant:
        # All layers should now have the minimum scale
        layers = [getattr(module, name) for name in layer_names]
        min_scale = layers[0].weight_global_scale.data
        for i, layer in enumerate(layers):
            if (layer_scale := layer.weight_global_scale.data) < min_scale:
                min_scale = layer_scale
        for layer in layers:
            assert torch.allclose(
                layer.weight_global_scale.data, min_scale
            ), f"Expected all layers to have scale {min_scale}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_class,layer_names",
    [
        (MockAttentionModule, ["q_proj", "k_proj", "v_proj"]),
        (MockMLPModule, ["gate_proj", "up_proj"]),
    ],
)
def test_fusion_preserves_effective_quantization(module_class, layer_names):
    """
    Test that fusing global_scale preserves the effective quantization by
    adjusting weight_scale proportionally.

    Verifies: full_scale = weight_scale / global_scale is unchanged
    """
    module = module_class(tensor_group_quant=True)

    # Set different global_scales and weight_scales for each layer
    layers = [getattr(module, name) for name in layer_names]
    for i, layer in enumerate(layers):
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor([1.0 + i * 0.5]), requires_grad=False
        )
        # Create weight_scale matching the weight shape for TENSOR_GROUP
        # For TENSOR_GROUP with group_size, shape is (out_features, num_groups)
        num_groups = layer.weight.shape[1] // 128  # group_size=128
        layer.weight_scale = torch.nn.Parameter(
            torch.full((layer.weight.shape[0], num_groups), 2.0 + i * 0.3),
            requires_grad=False,
        )

    # Compute effective full_scale before fusion
    full_scales_before = []
    for layer in layers:
        full_scale = layer.weight_scale / layer.weight_global_scale
        full_scales_before.append(full_scale.clone())

    # Fuse global scales
    update_fused_layer_weight_global_scales(module)

    # Verify effective full_scale is preserved
    for i, layer in enumerate(layers):
        full_scale_after = layer.weight_scale / layer.weight_global_scale
        assert torch.allclose(
            full_scale_after, full_scales_before[i], rtol=1e-5
        ), (
            f"Layer {layer_names[i]}: effective quantization changed after fusion. "
            f"Before: {full_scales_before[i][0, 0].item():.6f}, "
            f"After: {full_scale_after[0, 0].item():.6f}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_class,layer_names",
    [
        (MockAttentionModule, ["q_proj", "k_proj", "v_proj"]),
        (MockMLPModule, ["gate_proj", "up_proj"]),
    ],
)
def test_fusion_preserves_forward_output(module_class, layer_names):
    """
    Test that fusing global_scale doesn't change the forward pass output.

    Since fusion adjusts weight_scale to preserve full_scale = weight_scale / global_scale,
    the quantized output should be identical before and after fusion.
    """
    module = module_class(tensor_group_quant=True)
    layers = [getattr(module, name) for name in layer_names]

    # Set different global_scales and weight_scales
    for i, layer in enumerate(layers):
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor([1.0 + i * 0.5]), requires_grad=False
        )
        num_groups = layer.weight.shape[1] // 128
        layer.weight_scale = torch.nn.Parameter(
            torch.full((layer.weight.shape[0], num_groups), 2.0 + i * 0.3),
            requires_grad=False,
        )
        # Set some non-random weights for reproducibility
        torch.manual_seed(42 + i)
        layer.weight.data = torch.randn_like(layer.weight)

    # Compute outputs before fusion
    torch.manual_seed(123)
    inputs = [torch.randn(4, 128) for _ in layers]
    outputs_before = [layer(inp) for layer, inp in zip(layers, inputs)]

    # Fuse global scales
    update_fused_layer_weight_global_scales(module)

    # Compute outputs after fusion (same inputs)
    outputs_after = [layer(inp) for layer, inp in zip(layers, inputs)]

    # Verify outputs are identical
    for i, (before, after) in enumerate(zip(outputs_before, outputs_after)):
        assert torch.allclose(before, after, rtol=1e-5, atol=1e-7), (
            f"Layer {layer_names[i]}: forward output changed after fusion. "
            f"Max diff: {(before - after).abs().max().item():.6e}"
        )
