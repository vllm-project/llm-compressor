import copy
from types import MethodType
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.quantization import QuantizationArgs
from transformers import AutoModelForCausalLM, Qwen3MoeConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.observers.base import Observer
from llmcompressor.pipelines.sequential.offload import (
    subgraph_offload_modules,
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from llmcompressor.pipelines.sequential.restructure import RestructureHandler


@pytest.mark.parametrize(
    "targets,lazy_linearization,expected_lazy",
    [
        (None, None, True),
        (["ExpertMLP"], True, False),
        (["Linear"], True, False),
        (["model.layers.0.mlp.experts.0"], True, False),
        ([r"re:model\.layers\.0\.mlp\.experts\.0\..*_proj$"], True, False),
        (["Qwen3MoeDecoderLayer"], True, True),
        (["LinearExperts2D"], True, True),
        (["Qwen3MoeDecoderLayer", "ExpertMLP"], True, True),
        ([""], True, True),
        (["Qwen3MoeDecoderLayer"], False, False),
    ],
)
def test_lazy_linearization_option_and_split_target_fallback(
    monkeypatch, tmp_path, targets, lazy_linearization, expected_lazy
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
    config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    oneshot = Oneshot(
        model=model,
        processor=object(),
        recipe=[],
        sequential_targets=targets,
        **(
            {"lazy_linearization": lazy_linearization}
            if lazy_linearization is not None
            else {}
        ),
    )
    assert oneshot.dataset_args.lazy_linearization is expected_lazy
    assert all(not tensor.is_meta for tensor in model.parameters())
    with patch(
        "llmcompressor.entrypoints.oneshot.linearize_moe", wraps=linearize_moe
    ) as upfront_linearize:
        oneshot.apply_recipe_modifiers(None)
        assert upfront_linearize.call_count == (0 if expected_lazy else 1)
    for index, layer in enumerate(model.model.layers):
        torch.testing.assert_close(
            layer.mlp.experts.state_dict(),
            reference.model.layers[index].mlp.experts.state_dict(),
        )


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
    subgraph_onload_modules({"block.0": model.block[0]}, handler, device)
    torch.testing.assert_close(model.block[0].weight, initial)
    with torch.no_grad():
        model.block[0].weight.add_(1)
    subgraph_onload_modules({"other": model.other}, handler, device)
    torch.testing.assert_close(model.other.weight, initial + 1)
    assert model.block[0].weight is model.other.weight
    subgraph_onload_modules({"": model}, handler, device)
    assert model.position_alias is model.position
    handler.finish()
    # OffloadCache reads can create new tensors on the execution device.
    torch.testing.assert_close(model.position_alias.cpu(), torch.arange(4.0))
    torch.testing.assert_close(model.position.cpu(), torch.arange(4.0))
    assert not handler.mapping
    assert handler.source_root is None


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_source_caches_materialize_without_executing_source_forward(
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
    cache = model[0]._parameters
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    source = handler.mapping[model[0]]
    assert source._parameters is cache
    source.register_forward_pre_hook(
        lambda *_: pytest.fail("Retained sources should only supply tensor state")
    )
    calls = []
    model[0].register_forward_hook(lambda *args: calls.append("target"))
    assert source._forward_hooks is not model[0]._forward_hooks
    handler.configure_onload(torch.device("cpu"))
    if staged:
        subgraph_stage_modules({"0": model[0]}, handler)
        assert model[0].weight.is_meta
        assert cache.is_staged
    subgraph_onload_modules({"0": model[0]}, handler, torch.device("cpu"))
    torch.testing.assert_close(model[0](inputs), expected)
    assert calls == ["target"]
    assert not isinstance(model[0]._parameters, OffloadCache)
    assert not source._parameters  # Retained parents no longer keep the real tensors.
    handler.finish()


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_staging_shared_sources_keeps_targets_on_meta(tmp_path, offload_device):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model[0].register_buffer("position", torch.arange(4.0), persistent=False)
    expected = model[0].weight.detach().clone()
    offload_module(
        model[0],
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    handler.configure_onload(torch.device("cpu"))
    target = model[0]
    source = handler.mapping[target]
    weight = target.weight

    for _ in range(2):
        subgraph_stage_modules({"0": target, "alias": target}, handler)
        assert target.weight is weight
        assert target.weight.is_meta and target.position.is_meta
        assert handler.mapping[target] is source
        assert source._parameters.is_staged and source._buffers.is_staged
        assert source._parameters.offloaded_values["weight"].device.type == "cpu"

    subgraph_onload_modules({"0": target}, handler, torch.device("cpu"))
    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(target.position, torch.arange(4.0))
    handler.finish()


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_persistent_container_leaves_future_subgraph_sources_offloaded(
    tmp_path, offload_device
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    expected = model(torch.ones(2, 4)).detach()
    offload_module(
        model[0],
        onload_device=device,
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    handler.configure_onload(device)
    source = handler.mapping[model[0]]
    cache = source._parameters

    # Tracing can inline a container, leaving it outside its children's graphs.
    subgraph_stage_modules({"": model}, handler)
    assert not cache.is_staged
    subgraph_onload_modules({"": model}, handler, device)
    assert source._parameters is cache
    assert model[0].weight.is_meta

    modules = {"0": model[0], "1": model[1]}
    kwargs = subgraph_onload_modules(modules, handler, device)
    assert str(kwargs["0"]["offload_device"]) == offload_device
    assert kwargs["1"]["onload_device"] == device
    subgraph_offload_modules(modules, kwargs)
    torch.testing.assert_close(model(torch.ones(2, 4)).cpu(), expected)
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
    subgraph_onload_modules({"": model}, handler, device)
    assert model.weight_observer is observer
    assert observer._imatrix_count.item() == 0
    model(torch.ones(3, 4))
    assert observer._imatrix_count.item() == 3
    torch.testing.assert_close(observer._imatrix_sum, torch.full((4,), 3.0))
    handler.finish()


@pytest.mark.parametrize("patched_forward", [False, True])
@pytest.mark.parametrize("offload_state", ["none", "offloaded", "staged"])
def test_norm_restore_updates_same_mapping_before_offload(
    patched_forward, offload_state
):
    model = torch.nn.Module()
    model.norm = Gemma2RMSNorm(4, eps=1e-6)
    model.norm.weight = torch.nn.Parameter(torch.arange(4.0))
    if patched_forward:

        def forward(self, inputs):
            return inputs * (1 + self.weight)

        model.norm.forward = MethodType(forward, model.norm)
    model.alias = model.norm
    expected = model.norm.weight.detach().clone() + 2
    if offload_state != "none":
        offload_module(model.norm, onload_device="cpu", offload_device="cpu")
    handler = RestructureHandler(model)
    if offload_state == "staged":
        subgraph_stage_modules(dict(model.named_modules()), handler)
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
    offload_kwargs = subgraph_onload_modules(modules, handler, device)
    with torch.no_grad():
        target.weight.add_(2)
    handler.restore_restructured_modules(modules)
    assert model.norm is source
    assert model.alias is source
    assert modules["norm"] is source
    assert not isinstance(source._parameters, OffloadCache)
    torch.testing.assert_close(source.weight, expected)
    assert source.forward.__self__ is source
    inputs = torch.ones(2, 4)
    outputs = source(inputs)
    assert torch.isfinite(outputs).all()
    if patched_forward:
        torch.testing.assert_close(outputs, inputs * (1 + expected))
    assert not handler.restructures[2].mapping
    subgraph_offload_modules(modules, offload_kwargs)
    assert isinstance(source._parameters, OffloadCache)
    handler.finish()


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_finish_offloads_restored_norm_using_target_settings(tmp_path, offload_device):
    model = torch.nn.Module()
    model.norm = Gemma2RMSNorm(4, eps=1e-6)
    model.norm.weight = torch.nn.Parameter(torch.arange(4.0))
    offload_module(
        model.norm,
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    handler.configure_onload(torch.device("cpu"))
    target = model.norm
    modules = dict(model.named_modules())
    kwargs = subgraph_onload_modules(modules, handler, torch.device("cpu"))
    with torch.no_grad():
        target.weight.add_(2)
    subgraph_offload_modules(modules, kwargs)

    handler.finish()

    assert model.norm is not target
    assert isinstance(model.norm._parameters, OffloadCache)
    assert str(model.norm._parameters.offload_device) == offload_device
    if offload_device == "disk":
        assert model.norm._parameters.offload_dir == tmp_path
    torch.testing.assert_close(model.norm.weight, torch.arange(4.0) + 2)


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_finish_offloads_materialized_expert_children(tmp_path, offload_device):
    model = AutoModelForCausalLM.from_config(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=2,
            num_experts_per_tok=1,
        ),
        attn_implementation="eager",
    )
    reference = copy.deepcopy(model)
    linearize_moe(reference)
    for module in model.modules():
        offload_module(
            module,
            onload_device="cpu",
            offload_device=offload_device,
            **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
        )
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    experts = model.model.layers[0].mlp.experts
    assert experts in handler.mapping
    assert experts[0].gate_proj not in handler.mapping

    # Pipelines dispatch the prepared meta tree before onloading real sources.
    set_onload_device(model, torch.device("cpu"))
    handler.finish()

    for module in experts.modules():
        assert isinstance(module._parameters, OffloadCache)
        assert str(module._parameters.offload_device) == offload_device
    torch.testing.assert_close(
        {name: value.cpu() for name, value in model.state_dict().items()},
        {name: value.cpu() for name, value in reference.state_dict().items()},
        rtol=0,
        atol=0,
    )
