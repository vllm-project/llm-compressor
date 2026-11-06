"""
Unit tests for the expert pruning utilities shared by the pruning modifiers
(``llmcompressor.modifiers.pruning.expert_sparsity.utils``): structural MoE
layer pruning, which slices the expert module list and reorders the router.
"""

import pytest
import torch

from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.expert_sparsity.utils import prune_moe_layer
from tests.llmcompressor.modifiers.pruning.fake_moe import FakeMoEConfig, FakeMoEModel


@pytest.mark.unit
def test_prune_moe_layer_slices_experts_and_router():
    config = FakeMoEConfig(num_experts=8)
    model = FakeMoEModel(config)
    layer = model.layers[0]
    original_experts = [
        expert for expert in layer.experts.children() if isinstance(expert, ExpertMLP)
    ]
    original_router_weight = layer.gate.weight.detach().clone()

    retained = [1, 2, 4, 6, 7]
    assert prune_moe_layer("layers.0", layer, retained) == retained

    # retained experts are kept, in order, as the same module objects
    pruned_experts = [
        expert for expert in layer.experts.children() if isinstance(expert, ExpertMLP)
    ]
    assert len(pruned_experts) == 5
    for new_idx, original_idx in enumerate(retained):
        assert pruned_experts[new_idx] is original_experts[original_idx]

    # the router is sliced to the retained experts, and every expert-count
    # attribute in the layer is updated
    torch.testing.assert_close(layer.gate.weight, original_router_weight[retained])
    assert layer.gate.out_features == 5
    assert layer.gate.num_experts == 5
    assert layer.experts.num_experts == 5
    assert layer.num_experts == 5


@pytest.mark.unit
@pytest.mark.parametrize("bias_owner", ["router", "moe_block"])
def test_prune_moe_layer_resizes_correction_bias(bias_owner):
    """Group-limited routers carry a per-expert score-correction bias, either
    on the router itself or on the MoE block (e.g. Hy3); it is sliced along
    with the router's experts."""
    config = FakeMoEConfig(num_experts=8)
    model = FakeMoEModel(config)
    layer = model.layers[0]
    bias_holder = layer.gate if bias_owner == "router" else layer
    bias_holder.register_buffer(
        "e_score_correction_bias", torch.arange(8, dtype=torch.float32)
    )

    retained = [1, 2, 3, 5, 6, 7]
    prune_moe_layer("layers.0", layer, retained)

    torch.testing.assert_close(
        bias_holder.e_score_correction_bias,
        torch.tensor(retained, dtype=torch.float32),
    )
