from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.nn import functional as F
from torch.utils.data import DataLoader

from llmcompressor.core import active_session
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    get_pending_moe_onload_modules,
    materialize_pending_moe_weights,
    materialize_pending_moe_weights_immediately,
    set_onload_device_for_pending_moes,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.utils.dev import get_main_device
from tests.testing_utils import requires_gpu


class DummyLinearExperts(LinearExperts2D):
    is_concatenated = True
    is_transposed = False
    has_bias = False
    has_gate = True
    _apply_gate = staticmethod(
        lambda x: F.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]
    )


class SourceExperts(torch.nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=3, device="cpu"):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(2, 2 * intermediate_size, hidden_size, device=device)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(2, hidden_size, intermediate_size, device=device)
        )
        self.has_bias = False
        self.is_transposed = False


class Model(torch.nn.Module):
    def __init__(self, experts, device="cpu"):
        super().__init__()
        self.config = SimpleNamespace(model_type="test")
        self.experts = experts
        self._device = torch.device(device)

    @property
    def device(self):
        return self._device

    def forward(self, x, ids, weights):
        return self.experts(x, ids, weights)


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


@requires_gpu
def test_sequential_pipeline_initializes_modifiers_on_meta_and_materializes_onload():
    device = get_main_device()
    hidden_size = 128
    intermediate_size = 128
    config = SimpleNamespace(
        model_type="test",
        num_local_experts=2,
        num_experts_per_tok=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        use_bias=False,
        hidden_act="silu",
        swiglu_limit=None,
        dtype=torch.float32,
    )
    source = SourceExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    experts = DummyLinearExperts.meta_from_experts_module(source, config)
    model = Model(experts, device=device)

    weight_args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, group_size=128, strategy="group"
    )
    input_args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, strategy="tensor", dynamic=False
    )
    modifier = QuantizationModifier(
        config_groups={
            "toy": QuantizationScheme(
                targets=["Linear"],
                weights=weight_args,
                input_activations=input_args,
            )
        }
    )

    session = active_session()
    session.reset()
    try:
        session.initialize(
            model=model,
            recipe=[modifier],
            start=-1,
            pipeline="sequential",
        )
        linears = [
            module for module in model.modules() if isinstance(module, torch.nn.Linear)
        ]
        assert len(linears) == 6
        assert all(module.weight.is_meta for module in linears)
        assert all(module.weight_scale.is_meta for module in linears)

        hook_calls = []
        linears[0].register_forward_pre_hook(lambda *_: hook_calls.append(True))
        dataloader = DataLoader(
            [
                {
                    "x": torch.randn(4, hidden_size),
                    "ids": torch.tensor([[0], [1], [0], [1]]),
                    "weights": torch.ones(4, 1),
                }
            ],
            batch_size=None,
        )
        dataset_args = SimpleNamespace(
            sequential_offload_device="cpu",
            sequential_targets=["experts"],
            tracing_ignore=[],
            sequential_targets_per_subgraph=1,
            propagate_error=False,
            log_sequential_error=False,
            layerwise_decompression=False,
            layerwise_compression=False,
            stage_weights_in_pinned_memory=False,
            use_loss_mask=False,
            gpu_memory_utilization=0.9,
        )

        SequentialPipeline()(model, dataloader, dataset_args)

        assert hook_calls
        assert all(not module.weight.is_meta for module in linears)
        assert all(torch.isfinite(module.weight_scale).all() for module in linears)
        assert all(torch.isfinite(module.input_scale).all() for module in linears)
        session.finalize()
    finally:
        session.reset()
