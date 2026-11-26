from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.quantization import QuantizationArgs
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.core import LifecycleCallbacks
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import observe
from llmcompressor.observers import Observer
from llmcompressor.observers.fusion import FusionHandler
from llmcompressor.pipelines.sequential import pipeline as pipeline_module
from llmcompressor.pipelines.sequential.offload import (
    _submit_subgraph_staging,
    get_fused_weight_dependencies,
    materialize_modules,
    subgraph_offload_modules,
    subgraph_stage_modules,
)
from llmcompressor.pipelines.sequential.restructure import RestructureHandler


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_fused_dependencies_preserve_offload_and_observers(tmp_path, offload_device):
    device = torch.device("cpu")
    model = torch.nn.ModuleDict(
        {name: torch.nn.Linear(16, 16, bias=False) for name in ("gate", "up", "other")}
    )
    expected = model["up"].weight.detach().clone()
    for module in model.values():
        offload_module(
            module,
            onload_device=device,
            offload_device=offload_device,
            **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
        )
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    for name in ("gate", "up"):
        model[name].weight_observer = Observer.load_from_registry(
            "memoryless_minmax",
            base_name="weight",
            args=QuantizationArgs(num_bits=4, strategy="tensor_group", group_size=16),
        )
    FusionHandler.fuse(
        [(model[name].weight_observer, model[name]) for name in ("gate", "up")]
    )
    observer = model["up"].weight_observer
    module_names = {module: name for name, module in model.named_modules()}
    current = {"gate": model["gate"]}
    dependencies = get_fused_weight_dependencies(current, handler, module_names)
    assert dependencies == {"up": model["up"]}

    # Observer statistics can differ by rank; loading dependencies must not.
    observer.min_vals = torch.zeros(1)
    observer.max_vals = torch.ones(1)
    assert get_fused_weight_dependencies(current, handler, module_names) == dependencies
    observer.delete_statistics(check_fused=False)

    modules_to_materialize = current | dependencies
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            _submit_subgraph_staging(
                executor, dependencies, modules_to_materialize, handler, False
            )
            is None
        )

    kwargs = materialize_modules(modules_to_materialize, handler, device)
    observe(model["gate"], "weight")
    assert observer.has_statistics
    assert model["other"].weight.is_meta
    statistics = observer.get_statistics()
    subgraph_offload_modules(dependencies, kwargs)
    cache = model["up"]._parameters
    assert isinstance(cache, OffloadCache)
    assert str(cache.offload_device) == offload_device
    if offload_device == "disk":
        assert cache.offload_dir == tmp_path
    assert get_fused_weight_dependencies(current, handler, module_names) == {}

    # Its own subgraph uses the ordinary staging/onload path and the same observer.
    subgraph_stage_modules(dependencies, handler)
    materialize_modules(dependencies, handler, device)
    assert model["up"].weight_observer is observer
    assert all(getattr(observer, name) is value for name, value in statistics.items())
    torch.testing.assert_close(model["up"].weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("modifier_cls", [QuantizationModifier, GPTQModifier])
@pytest.mark.parametrize("weight_observer", ["memoryless_minmax", "imatrix_mse"])
@pytest.mark.parametrize("layerwise_compression", [False, True])
def test_linear_targets_observe_future_fused_weights(
    tmp_path, monkeypatch, modifier_cls, weight_observer, layerwise_compression
):
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.config.name_or_path = str(tmp_path)
    gate, up = model.model.layers[0].mlp.gate_proj, model.model.layers[0].mlp.up_proj
    original_up = up.weight.detach().clone()
    epoch_end = LifecycleCallbacks.sequential_epoch_end
    offload = pipeline_module.subgraph_offload_modules
    events = []

    def check_epoch_end(modules):
        if gate in modules.values():
            assert up not in modules.values()
            assert not up.weight.is_meta
            torch.testing.assert_close(up.weight.cpu(), original_up, rtol=0, atol=0)
            events.append("gate")
        if up in modules.values():
            assert events == ["gate", "future_offload"]
            events.append("up")
        epoch_end(modules)

    def check_offload(modules, kwargs):
        offload(modules, kwargs)
        if set(modules.values()) == {up} and events == ["gate"]:
            assert isinstance(up._parameters, OffloadCache)
            events.append("future_offload")

    monkeypatch.setattr(LifecycleCallbacks, "sequential_epoch_end", check_epoch_end)
    monkeypatch.setattr(pipeline_module, "subgraph_offload_modules", check_offload)
    data = DataLoader(
        [{"input_ids": torch.randint(0, 32, (1, 8))} for _ in range(4)],
        batch_size=None,
    )
    oneshot(
        model=model,
        processor=object(),
        dataset=data,
        recipe=modifier_cls(
            targets="Linear",
            scheme="NVFP4A16",
            ignore=["lm_head"],
            weight_observer=weight_observer,
        ),
        pipeline="sequential",
        sequential_targets=["Linear"],
        sequential_targets_per_subgraph=1,
        stage_weights_in_pinned_memory=False,
        layerwise_compression=layerwise_compression,
    )
    assert events == ["gate", "future_offload", "up"]
    assert all(not tensor.is_meta for tensor in model.state_dict().values())
    torch.testing.assert_close(gate.weight_global_scale, up.weight_global_scale)
