from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from torch.nn import functional as F

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    get_pending_moe_onload_modules,
    materialize_pending_moe_weights,
    materialize_pending_moe_weights_immediately,
    set_onload_device_for_pending_moes,
)


class DummyLinearExperts(LinearExperts2D):
    is_concatenated = True
    is_transposed = False
    has_bias = False
    has_gate = True
    _apply_gate = staticmethod(lambda x: F.silu(x[..., :3]) * x[..., 3:])


class SourceExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 6, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 3))
        self.has_bias = False
        self.is_transposed = False


class Model(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.config = SimpleNamespace(model_type="test")
        self.experts = experts


def test_deferred_linearization_materializes_after_onload():
    config = SimpleNamespace(
        model_type="test",
        num_local_experts=2,
        num_experts_per_tok=1,
        hidden_size=4,
        intermediate_size=3,
        use_bias=False,
        hidden_act="silu",
        swiglu_limit=None,
        dtype=torch.float32,
    )
    source = SourceExperts()
    source_gate_up = source.gate_up_proj.detach().clone()
    experts = DummyLinearExperts.meta_from_experts_module(source, config)
    model = Model(experts)

    gate_weight = experts[0].gate_proj.weight
    probe = torch.tensor([17.0])
    experts.register_buffer("observer_probe", probe)
    experts[0].gate_proj.register_parameter(
        "weight_zero_point", torch.nn.Parameter(torch.zeros(1, device="meta"))
    )
    hook_observations = []
    experts[0].gate_proj.register_forward_pre_hook(
        lambda module, _: hook_observations.append(module.weight is gate_weight)
    )

    modules = {
        "experts" if not name else f"experts.{name}": module
        for name, module in experts.named_modules()
    }
    set_onload_device_for_pending_moes(model, "cpu")
    staged_modules = get_pending_moe_onload_modules(modules)
    offload_kwargs = subgraph_onload_modules(staged_modules)
    materialize_pending_moe_weights(modules, offload_kwargs)

    assert experts[0].gate_proj.weight is gate_weight
    assert torch.equal(experts[0].gate_proj.weight, source_gate_up[0, :3])
    assert torch.equal(experts[0].up_proj.weight, source_gate_up[0, 3:])
    assert torch.equal(experts[0].gate_proj.weight_zero_point, torch.zeros(1))
    assert experts.observer_probe is probe
    assert torch.equal(experts.observer_probe, torch.tensor([17.0]))
    assert not hasattr(experts, "_pending_source_experts")

    subgraph_offload_modules(modules, offload_kwargs)
    experts[0].gate_proj(torch.randn(2, 4))
    assert hook_observations == [True]


def test_deferred_linearization_validates_projection_shapes_on_meta():
    config = SimpleNamespace(
        model_type="test",
        num_local_experts=2,
        num_experts_per_tok=1,
        hidden_size=4,
        intermediate_size=3,
        use_bias=False,
        hidden_act="silu",
        swiglu_limit=None,
        dtype=torch.float32,
    )
    source = SourceExperts()
    source.gate_up_proj = torch.nn.Parameter(torch.randn(2, 6, 5))

    with pytest.raises(RuntimeError, match="size of tensor"):
        DummyLinearExperts.meta_from_experts_module(source, config)


def test_nonsequential_materialization_completes_pending_moes_immediately():
    config = SimpleNamespace(
        model_type="test",
        num_local_experts=2,
        num_experts_per_tok=1,
        hidden_size=4,
        intermediate_size=3,
        use_bias=False,
        hidden_act="silu",
        swiglu_limit=None,
        dtype=torch.float32,
    )
    source = SourceExperts()
    source_gate_up = source.gate_up_proj.detach().clone()
    experts = DummyLinearExperts.meta_from_experts_module(source, config)
    model = Model(experts)

    materialize_pending_moe_weights_immediately(model)

    assert not hasattr(experts, "_pending_source_experts")
    assert torch.equal(experts[0].gate_proj.weight, source_gate_up[0, :3])
