import pytest

from llmcompressor.modeling.moe.linearize import (
    assign_child_offload_kwargs,
    get_non_linearized_moes,
    linearize_moe,
    linearize_onload_experts,
)


class _NamedModules:
    def submodule_dict(self, model):
        return dict(model.named_modules())


def test_assign_child_offload_kwargs_copies_parent_settings():
    offload_kwargs = {"model.layers.0.mlp.experts": {"offload_device": "cpu"}}
    assign_child_offload_kwargs(
        ["model.layers.0.mlp.experts"],
        [
            "model.layers.0.mlp.experts",
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.0.self_attn.q_proj",
        ],
        offload_kwargs,
    )
    assert offload_kwargs["model.layers.0.mlp.experts.0.down_proj"] == {
        "offload_device": "cpu"
    }
    assert "model.layers.0.self_attn.q_proj" not in offload_kwargs


def test_linearize_onload_experts_replaces_only_the_named_experts():
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    from torch import nn
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeExperts,
    )

    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.utils.dev import skip_weights_initialize

    config = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=1,
        vocab_size=32,
    )

    class _Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            with skip_weights_initialize():
                self.experts = Qwen3MoeExperts(config)
            self.config = config

    model = _Holder()
    modules = {"experts": model.experts}
    roots = linearize_onload_experts(model, _NamedModules(), modules, {})
    assert len(roots) == 1
    assert isinstance(model.experts, LinearExperts2D)
    assert model.experts[0].down_proj.weight.ndim == 2
    assert get_non_linearized_moes(model) == []


def _qwen_config():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=1,
        vocab_size=32,
        max_position_embeddings=32,
    )


def test_offloaded_load_stays_fused_until_onload():
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    import torch
    from compressed_tensors.offload import get_offloaded_device, offload_module
    from compressed_tensors.offload.module import (
        subgraph_offload_modules,
        subgraph_onload_modules,
    )
    from torch import nn
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from llmcompressor.core import active_session
    from llmcompressor.modeling.moe import linearize as linearize_module
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.modeling.moe.linearize import (
        linearize_onload_experts,
        load_quantizable_moe,
    )
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.quantization.quantization.mixin import (
        adopt_new_modules,
    )
    from llmcompressor.utils.dev import skip_weights_initialize

    config = _qwen_config()

    class _Config:
        model_type = "not-a-mapped-moe"

    class _Model(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = cls()
            with skip_weights_initialize():
                model.experts = Qwen3MoeExperts(config)
            model.config = config
            with torch.no_grad():
                model.experts.down_proj.fill_(1)
                model.experts.gate_up_proj.fill_(1)
            if kwargs.get("offload"):
                offload_module(
                    model.experts,
                    onload_device=torch.device("cpu"),
                    offload_device=torch.device("cpu"),
                )
            return model

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            linearize_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: _Config(),
        )
        with load_quantizable_moe(_Model):
            offloaded = _Model.from_pretrained("unused", offload=True)
            resident = _Model.from_pretrained("unused", offload=False)

    assert isinstance(offloaded.experts, Qwen3MoeExperts)
    assert isinstance(resident.experts, LinearExperts2D)

    session = active_session()
    session.reset()
    try:
        session.initialize(
            model=offloaded,
            recipe=QuantizationModifier(targets=["Linear"], scheme="W8A8"),
        )
        modules = {"experts": offloaded.experts}
        offload_kwargs = subgraph_onload_modules(modules)
        assert offloaded.experts.down_proj.device.type == "cpu"
        adopt_new_modules(
            linearize_onload_experts(
                offloaded, _NamedModules(), modules, offload_kwargs
            ),
            start_observers=True,
        )
        down = offloaded.experts[0].down_proj
        assert torch.equal(down.weight, torch.ones_like(down.weight))
        assert down.quantization_scheme.weights.num_bits == 8
        assert down.quantization_scheme.weights.strategy == "channel"
        assert type(down.weight_observer).__name__ == "MemorylessMinMaxObserver"
        subgraph_offload_modules(modules, offload_kwargs)
        assert get_offloaded_device(down) == torch.device("cpu")
        assert isinstance(offloaded.experts, LinearExperts2D)
    finally:
        session.reset()


def test_fused_trace_keeps_moe_block_callable():
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    import torch
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM,
    )

    from llmcompressor.args.dataset_arguments import DatasetArguments
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
    from llmcompressor.utils.dev import skip_weights_initialize

    sample = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    ignore = DatasetArguments().tracing_ignore
    for targets in (["Linear"], ["ExpertMLP"]):
        config = _qwen_config()
        with skip_weights_initialize():
            model = Qwen3MoeForCausalLM(config)
        subgraphs = trace_subgraphs(model, sample, targets, ignore)
        owners = [
            subgraph
            for subgraph in subgraphs
            if "model.layers.0.mlp.experts" in subgraph.submodule_dict(model)
        ]
        assert len(owners) == 1
        assert any(
            node.target == "model.layers.0.mlp"
            for node in owners[0].graph.find_nodes(op="call_module")
        )
        block = model.get_submodule("model.layers.0.mlp")
        linearize_moe(model)
        hidden = torch.ones(1, 4, config.hidden_size)
        output = block(hidden)
        assert tuple(output.shape) == (1, 4, config.hidden_size)
        assert isinstance(block.experts, LinearExperts2D)


def test_sequential_onload_linearizes_offloaded_qwen_experts():
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    import torch
    from compressed_tensors.offload import get_offloaded_device, offload_module
    from compressed_tensors.quantization import QuantizationStatus
    from torch.utils.data import DataLoader
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    from llmcompressor.args.dataset_arguments import DatasetArguments
    from llmcompressor.core import active_session
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
    from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
    from llmcompressor.utils.dev import skip_weights_initialize

    with skip_weights_initialize():
        model = Qwen3MoeForCausalLM(_qwen_config())
    experts = model.model.layers[0].mlp.experts
    with torch.no_grad():
        experts.down_proj.fill_(1)
        experts.gate_up_proj.fill_(1)
    offload_module(
        experts,
        onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    assert type(experts).__name__ == "Qwen3MoeExperts"

    # Record the class at onload. Linearize replaces the module after that call.
    onload_types = []
    real_onload = sequential_pipeline.subgraph_onload_modules

    def record_onload(modules):
        for name, module in modules.items():
            if name.endswith("mlp.experts"):
                onload_types.append(type(module).__name__)
        return real_onload(modules)

    dataloader = DataLoader(
        [{"input_ids": torch.zeros(1, 4, dtype=torch.long)}],
        batch_size=1,
        collate_fn=lambda batch: batch[0],
    )
    session = active_session()
    session.reset()
    try:
        session.initialize(
            model=model,
            recipe=QuantizationModifier(targets=["Linear"], scheme="W8A8"),
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                sequential_pipeline, "subgraph_onload_modules", record_onload
            )
            SequentialPipeline()(
                model,
                dataloader,
                DatasetArguments(propagate_error=False, log_sequential_error=False),
            )
        assert onload_types == ["Qwen3MoeExperts"]
        experts = model.model.layers[0].mlp.experts
        down_proj = experts[0].down_proj
        assert isinstance(experts, LinearExperts2D)
        assert torch.equal(down_proj.weight, torch.ones_like(down_proj.weight))
        assert get_offloaded_device(down_proj) == torch.device("cpu")
        assert down_proj.quantization_scheme.weights.num_bits == 8
        assert down_proj.quantization_scheme.weights.strategy == "channel"
        assert down_proj.quantization_status == QuantizationStatus.FROZEN
        # Exact 1/127 fails on MPS. The stored scale is one step away.
        scale = down_proj.weight_scale.detach().float().cpu()
        expected = torch.full_like(scale, 1 / 127)
        assert torch.allclose(scale, expected, rtol=0, atol=1e-4)
    finally:
        session.reset()
