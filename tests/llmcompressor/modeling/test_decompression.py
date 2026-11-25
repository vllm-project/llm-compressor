import copy

import pytest
import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3MoeConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modeling.decompression import DecompressionRestructure
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modeling.offset_norm import NormCalibrationModule
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.offload import (
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from llmcompressor.pipelines.sequential.restructure import RestructureHandler
from llmcompressor.utils.helpers import DisableQuantization


def _compress(layer, compression_format="pack-quantized"):
    scheme = QuantizationScheme(
        targets=["Linear"],
        format=compression_format,
        weights=QuantizationArgs(
            num_bits=4 if compression_format == "pack-quantized" else 8,
            strategy="channel",
        ),
    )
    initialize_module_for_quantization(layer, scheme)
    with torch.no_grad():
        layer.weight_scale.fill_(0.0625)
        layer.weight_zero_point.zero_()
    layer.quantization_status = QuantizationStatus.FROZEN
    compress_module(layer)


@pytest.mark.parametrize("already_decompressed", [False, True])
def test_restructuring_requires_decompression_opt_in(already_decompressed):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    _compress(model[1])
    if already_decompressed:
        decompress_module(model[1], leave_decompressed=True)
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    source = handler.mapping[model[1]]["source"]

    with pytest.raises(ValueError, match="Module '1'.*layerwise_decompression=True"):
        handler.apply_restructuring()

    assert all(parameter.is_meta for parameter in model.parameters())
    assert all(not parameter.is_meta for parameter in source.parameters())
    assert model[1].quantization_status == source.quantization_status


@pytest.mark.parametrize("remaining_compressed_module", [False, True])
def test_restructuring_checks_model_after_upfront_linearization(
    remaining_compressed_module,
):
    model = torch.nn.Module()
    model.config = Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    model.experts = Qwen3MoeExperts(model.config)
    model.experts.quantization_status = QuantizationStatus.DECOMPRESSED
    if remaining_compressed_module:
        model.other = torch.nn.Linear(16, 16)
        _compress(model.other)

    linearize_moe(model)
    assert not hasattr(model.experts, "quantization_status")
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    if remaining_compressed_module:
        with pytest.raises(ValueError, match="Module 'other'.*layerwise_decompression"):
            handler.apply_restructuring()
    else:
        handler.apply_restructuring()
        assert all(parameter.is_meta for parameter in model.parameters())


@pytest.mark.parametrize("compression_format", ["pack-quantized", "int-quantized"])
@pytest.mark.parametrize("already_decompressed", [False, True])
@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_decompression_preserves_new_recipe_state(
    tmp_path, compression_format, already_decompressed, offload_device
):
    # Packing pads thirteen columns; meta restructuring must retain the exact shape.
    layer = torch.nn.Linear(13, 5)
    _compress(layer, compression_format)
    reference = copy.deepcopy(layer)
    decompress_module(reference, leave_decompressed=False)
    if already_decompressed:
        decompress_module(layer, leave_decompressed=True)
    offload_module(
        layer,
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    model = torch.nn.Sequential(layer)
    set_onload_device(model, torch.device("cpu"))
    handler = RestructureHandler(model, layerwise_decompression=True)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    assert model[0] is layer
    assert layer.weight.is_meta
    assert layer.weight.shape == (5, 13)
    assert not hasattr(layer, "quantization_scheme")
    modifier = QuantizationModifier(
        config_groups={
            "new": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=8, strategy="channel"),
            )
        },
        weight_observer="memoryless_minmax",
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)
    scheme = layer.quantization_scheme
    assert DecompressionRestructure().apply(model) is model
    assert layer.quantization_scheme is scheme
    observer = layer.weight_observer
    weight = layer.weight
    calls = []
    layer.register_forward_pre_hook(lambda *_: calls.append("target"))

    subgraph_onload_modules({"0": layer}, handler, torch.device("cpu"))
    assert layer.quantization_scheme is scheme
    assert layer.weight_observer is observer
    assert layer.weight is weight
    torch.testing.assert_close(layer.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias, reference.bias, rtol=0, atol=0)
    inputs = torch.randn(2, 13)
    with DisableQuantization(model):
        torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
    assert calls == ["target"]
    modifier.on_calibration_end(state, None)


@pytest.mark.parametrize(
    "modifier_class,pipeline,recompress",
    [
        (QuantizationModifier, "basic", False),
        (QuantizationModifier, "datafree", False),
        (QuantizationModifier, "sequential", True),
        (GPTQModifier, "sequential", True),
    ],
)
def test_oneshot_recalibrates_packed_inputs_with_decompression_enabled(
    tmp_path, modifier_class, pipeline, recompress
):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.config.name_or_path = str(tmp_path)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            _compress(layer)
    reference = copy.deepcopy(model)
    DecompressionRestructure(enabled=True).apply(reference)
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
        {"input_ids": torch.tensor([[5, 6, 7, 8]])},
    ]
    for candidate in (reference, model):
        recipe = modifier_class(
            config_groups={
                "new": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(num_bits=8, strategy="channel"),
                )
            },
            ignore=["lm_head"],
            weight_observer="memoryless_minmax",
        )
        oneshot(
            model=candidate,
            processor=object(),
            recipe=recipe,
            dataset=None
            if pipeline == "datafree"
            else DataLoader(batches, batch_size=None),
            pipeline=pipeline,
            sequential_targets=["LlamaDecoderLayer"],
            layerwise_decompression=True,
            layerwise_compression=recompress,
        )
        assert not any(
            hasattr(module, "weight_observer") for module in candidate.modules()
        )
        assert all(not tensor.is_meta for tensor in candidate.state_dict().values())
    torch.testing.assert_close(
        model.state_dict(), reference.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("layerwise_decompression", [False, True])
def test_decompression_flag_does_not_skip_model_validation(
    tmp_path, layerwise_decompression
):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.quantization_config = {"quant_method": "gptq"}
    config.save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_config(config)
    model.config.name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="different format"):
        oneshot(
            model=model,
            processor=object(),
            recipe=[],
            layerwise_decompression=layerwise_decompression,
        )


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
@pytest.mark.parametrize("stage", [False, True])
def test_linearization_decompresses_its_source_tree(
    monkeypatch, tmp_path, offload_device, stage
):
    config = Qwen3MoeConfig(
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
        dtype="float32",
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    original = model.model.layers[0].mlp.experts
    experts_class = LinearExperts2D.get_linear_experts_cls(type(original))

    class PackedExperts(torch.nn.Module):
        is_transposed = False
        has_bias = False

        def __init__(self):
            super().__init__()
            self.gate_up_storage = torch.nn.Linear(16, 64, bias=False)
            self.down_storage = torch.nn.Linear(16, 32, bias=False)
            with torch.no_grad():
                self.gate_up_storage.weight.copy_(original.gate_up_proj.flatten(0, 1))
                self.down_storage.weight.copy_(original.down_proj.flatten(0, 1))

        @property
        def gate_up_proj(self):
            return self.gate_up_storage.weight.view(2, 32, 16)

        @property
        def down_proj(self):
            return self.down_storage.weight.view(2, 16, 16)

    monkeypatch.setitem(LinearExperts2D._registry, PackedExperts, experts_class)
    packed = PackedExperts()
    for module in packed.children():
        _compress(module)
    model.model.layers[0].mlp.experts = packed
    reference = copy.deepcopy(model)
    DecompressionRestructure(enabled=True).apply(reference)
    linearize_moe(reference)
    expected = reference.model.layers[0].mlp.experts
    for module in packed.children():
        offload_module(
            module,
            onload_device="cpu",
            offload_device=offload_device,
            **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
        )

    set_onload_device(model, torch.device("cpu"))
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    old_children = list(packed.children())
    handler.apply_restructuring()
    target = model.model.layers[0].mlp.experts
    assert isinstance(target, LinearExperts2D)
    assert all(child not in handler.mapping for child in old_children)
    entry = handler.mapping[target]
    source = entry["source"]
    assert entry["restructures"] == [handler.restructures[0]]
    assert not entry["materialized"]
    assert "weight_packed" in source.gate_up_storage._parameters

    modifier = QuantizationModifier(
        config_groups={
            "new": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=8, strategy="channel"),
            )
        },
        weight_observer="memoryless_minmax",
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)
    projection = target[0].gate_proj
    observer = projection.weight_observer
    calls = []
    projection.register_forward_hook(lambda *_: calls.append("target"))
    if stage:
        subgraph_stage_modules({"experts": target}, handler)
        assert projection.weight.is_meta
        assert source.gate_up_storage._parameters.is_staged
    subgraph_onload_modules({"experts": target}, handler, torch.device("cpu"))
    assert target[0].gate_proj is projection
    assert projection.weight_observer is observer
    for name, weight in expected.named_parameters():
        torch.testing.assert_close(target.get_parameter(name), weight, rtol=0, atol=0)
    inputs = torch.randn(4, 16)
    routing = torch.tensor([[0], [1], [0], [1]])
    routing_weights = torch.ones(4, 1)
    torch.testing.assert_close(
        target(inputs, routing, routing_weights),
        expected(inputs, routing, routing_weights),
        rtol=0,
        atol=0,
    )
    assert calls
    assert not source.gate_up_storage._parameters
    assert target not in handler.mapping


def test_decompression_then_offset_norm_reuses_the_same_source():
    model = torch.nn.Module()
    model.norm = Gemma2RMSNorm(16, eps=1e-6)
    model.norm.weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.int8), requires_grad=False
    )
    model.norm.weight_scale = torch.nn.Parameter(
        torch.tensor([0.125]), requires_grad=False
    )
    model.norm.quantization_scheme = QuantizationScheme(
        targets=["Gemma2RMSNorm"],
        format="int-quantized",
        weights=QuantizationArgs(num_bits=8, strategy="tensor"),
    )
    model.norm.quantization_status = QuantizationStatus.COMPRESSED
    set_onload_device(model, torch.device("cpu"))
    handler = RestructureHandler(model, layerwise_decompression=True)
    handler.convert_model_to_meta()
    original = model.norm
    entry = handler.mapping[original]
    source = entry["source"]
    handler.apply_restructuring()
    target = model.norm
    assert isinstance(target, NormCalibrationModule)
    assert original not in handler.mapping
    assert handler.mapping[target] is entry
    assert entry["restructures"] == list(handler.restructures[1:])
    assert not entry["materialized"]
    modules = {"norm": target}
    subgraph_onload_modules(modules, handler, torch.device("cpu"))
    assert handler.mapping[target] is entry
    assert entry["materialized"]
    assert source._parameters  # Only the norm operation retains this source now.
    torch.testing.assert_close(source.weight, torch.arange(16) * 0.125)
    with torch.no_grad():
        target.weight.add_(2)
    # A retained restoration entry must not replay decompression or norm conversion.
    subgraph_onload_modules(modules, handler, torch.device("cpu"))
    torch.testing.assert_close(target.weight, torch.arange(16) * 0.125 + 3)
    handler.restore_restructured_modules(modules)
    assert model.norm is source
    assert modules["norm"] is source
    assert target not in handler.mapping
    torch.testing.assert_close(source.weight, torch.arange(16) * 0.125 + 2)
