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
