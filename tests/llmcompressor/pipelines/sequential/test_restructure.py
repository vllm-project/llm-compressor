import copy
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from compressed_tensors.quantization import QuantizationArgs
from transformers import AutoModelForCausalLM, Qwen3MoeConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.observers.base import Observer
from llmcompressor.pipelines.sequential.restructure import RestructureHandler


@pytest.mark.parametrize(
    "targets,eager_layers",
    [
        (["ExpertMLP"], {0, 1}),
        (["Linear"], {0, 1}),
        (["model.layers.0.mlp.experts.0"], {0}),
        ([r"re:model\.layers\.0\.mlp\.experts\.0\..*_proj$"], {0}),
        (["Qwen3MoeDecoderLayer"], set()),
        (["LinearExperts2D"], set()),
        (["Qwen3MoeDecoderLayer", "ExpertMLP"], set()),
        ([""], set()),
    ],
)
def test_split_moes_are_linearized_upfront_but_initialize_on_meta(
    monkeypatch, targets, eager_layers
):
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.restructure.get_main_device",
        lambda: torch.device("cpu"),
    )
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        dtype="float32",
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    reference = copy.deepcopy(model)
    linearize_moe(reference)
    oneshot = Oneshot.__new__(Oneshot)
    oneshot.model = model
    oneshot.dataset_args = SimpleNamespace(
        layerwise_decompression=False, sequential_targets=targets
    )
    oneshot._linearize_moe_if_needed()
    assert all(not tensor.is_meta for tensor in model.parameters())
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    assert not handler.restructures
    handler.apply_restructuring()
    assert all(tensor.is_meta for tensor in model.parameters())
    assert all(tensor.is_meta for tensor in model.buffers())
    for index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        assert (experts not in handler.restructures) == (index in eager_layers)
    handler.materialize_all()
    for index, layer in enumerate(model.model.layers):
        torch.testing.assert_close(
            layer.mlp.experts.state_dict(),
            reference.model.layers[index].mlp.experts.state_dict(),
        )
    handler.finish()
    assert not handler.mapping


def test_delay_preserves_shared_modules_parameters_and_tensor_attributes():
    model = torch.nn.Module()
    model.block = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model.alias = model.block
    model.other = torch.nn.Linear(4, 4, bias=False)
    model.other.weight = model.block[0].weight
    model.register_buffer("position", torch.arange(4.0), persistent=False)
    model.position_alias = model.position
    initial = model.block[0].weight.detach().clone()
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    assert model.block is model.alias
    assert model.block[0].weight is model.other.weight
    assert model.position_alias is model.position
    assert model.position.is_meta
    assert handler.source_root.block is handler.source_root.alias
    assert handler.mapping[model.block]._modules is not model.block._modules
    device = torch.device("cpu")
    handler.configure_onload(device)
    handler.onload_and_restructure_modules({"block.0": model.block[0]}, device)
    torch.testing.assert_close(model.block[0].weight, initial)
    with torch.no_grad():
        model.block[0].weight.add_(1)
    handler.onload_and_restructure_modules({"other": model.other}, device)
    torch.testing.assert_close(model.other.weight, initial + 1)
    assert model.block[0].weight is model.other.weight
    handler.finish()
    assert model.position_alias is model.position
    torch.testing.assert_close(model.position, torch.arange(4.0))
    assert not handler.mapping
    assert handler.source_root is None


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_source_dispatch_is_rebound_and_new_hooks_stay_on_target(
    staged, offload_device, tmp_path
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    inputs = torch.ones(2, 4)
    expected = model(inputs).detach()
    offload_module(
        model[0],
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    set_onload_device(model, torch.device("cpu"))
    if staged:
        subgraph_stage_modules(dict(model.named_modules()))
    cache = model[0]._parameters
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    source = handler.mapping[model[0]]
    assert source._parameters is cache
    assert source.forward.__self__ is source
    calls = []
    model[0].register_forward_hook(lambda *args: calls.append("target"))
    assert source._forward_hooks is not model[0]._forward_hooks
    subgraph_onload_modules({"0": source})
    torch.testing.assert_close(source(inputs), expected)
    assert not calls
    handler.configure_onload(torch.device("cpu"))
    handler.onload_and_restructure_modules({"0": model[0]}, torch.device("cpu"))
    torch.testing.assert_close(model[0](inputs), expected)
    assert calls == ["target"]
    assert not isinstance(model[0]._parameters, OffloadCache)
    assert not source._parameters  # Retained parents no longer keep the real tensors.
    handler.finish()


def test_new_observer_meta_buffers_materialize_with_ordinary_module():
    model = torch.nn.Linear(4, 4)
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    observer = Observer.load_from_registry(
        "imatrix_mse",
        base_name="weight",
        args=QuantizationArgs(num_bits=4, strategy="channel"),
    )
    model.register_module("weight_observer", observer)
    observer.attach(model)
    assert observer._imatrix_sum.is_meta
    device = torch.device("cpu")
    handler.configure_onload(device)
    handler.onload_and_restructure_modules({"": model}, device)
    assert model.weight_observer is observer
    assert observer._imatrix_count.item() == 0
    model(torch.ones(3, 4))
    assert observer._imatrix_count.item() == 3
    torch.testing.assert_close(observer._imatrix_sum, torch.full((4,), 3.0))
    handler.finish()


def test_norm_restore_updates_same_mapping_before_offload():
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.norm = Gemma2RMSNorm(4, eps=1e-6)
    model.norm.weight = torch.nn.Parameter(torch.arange(4.0))
    model.alias = model.norm
    expected = model.norm.weight.detach().clone() + 2
    handler = RestructureHandler(model)
    original_norm = model.norm
    handler.convert_model_to_meta()
    assert model.norm is original_norm
    assert original_norm.weight.is_meta
    handler.apply_restructuring()
    target = model.norm
    assert model.alias is target
    source = handler.mapping[target]
    device = torch.device("cpu")
    handler.configure_onload(device)
    modules = {"norm": target}
    handler.onload_and_restructure_modules(modules, device)
    with torch.no_grad():
        target.weight.add_(2)
    handler.restore_restructured_modules(modules, offload_sources=False)
    assert model.norm is source
    assert model.alias is source
    assert modules["norm"] is source
    torch.testing.assert_close(source.weight, expected)
    assert not handler.restore_sources
    handler.finish()
