import copy
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3MoeConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modeling.decompression import DecompressionRestructure
from llmcompressor.modeling.module_restructure import ModuleRestructure, replace_module
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modeling.offset_norm import NormCalibrationModule
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
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
    handler = RestructureHandler(model)
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

    handler.configure_onload(torch.device("cpu"))
    handler.onload_and_restructure_modules({"0": layer}, torch.device("cpu"))
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
    handler.finish()


@pytest.mark.parametrize(
    "modifier_class,pipeline,recompress",
    [
        (QuantizationModifier, "basic", False),
        (QuantizationModifier, "datafree", False),
        (QuantizationModifier, "sequential", True),
        (GPTQModifier, "sequential", True),
    ],
)
def test_oneshot_recalibrates_packed_inputs_automatically(
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
    DecompressionRestructure().apply(reference)
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
            layerwise_compression=recompress,
        )
        assert not any(
            hasattr(module, "weight_observer") for module in candidate.modules()
        )
        assert all(not tensor.is_meta for tensor in candidate.state_dict().values())
    torch.testing.assert_close(
        model.state_dict(), reference.state_dict(), rtol=0, atol=0
    )


def test_automatic_decompression_keeps_unsupported_format_validation(tmp_path):
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
    model = AutoModelForCausalLM.from_config(config)
    model.config.name_or_path = str(tmp_path)
    model.config.quantization_config = {"quant_method": "gptq"}
    with pytest.raises(ValueError, match="different format"):
        oneshot(model=model, processor=object(), recipe=[])


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_decompression_then_linearization_materializes_one_source_tree(
    monkeypatch, tmp_path, offload_device
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
    DecompressionRestructure().apply(reference)
    linearize_moe(reference)
    expected = reference.model.layers[0].mlp.experts
    for module in packed.children():
        offload_module(
            module,
            onload_device="cpu",
            offload_device=offload_device,
            **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
        )

    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    old_children = list(packed.children())
    handler.apply_restructuring()
    target = model.model.layers[0].mlp.experts
    assert isinstance(target, LinearExperts2D)
    assert handler.restructured_modules == {target}
    assert all(child not in handler.mapping for child in old_children)
    source = handler.mapping[target]
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
    handler.configure_onload(torch.device("cpu"))
    handler.onload_and_restructure_modules({"experts": target}, torch.device("cpu"))
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
    handler.finish()


def test_parent_replacement_retains_nested_norm_for_restoration():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
            self.norm = Gemma2RMSNorm(16, eps=1e-6)

    class WrapBlock(ModuleRestructure):
        def apply(self, model, on_restructure=None):
            for name, source in list(model.named_modules()):
                if isinstance(source, Block):
                    target = torch.nn.Sequential(source.linear, source.norm)
                    if on_restructure is not None:
                        on_restructure(name, source, target)
                    model = replace_module(model, source, target)
            return model

    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.block = Block()
    _compress(model.block.linear)
    handler = RestructureHandler(model)
    handler.restructures = (
        handler.restructures[0],
        WrapBlock(),
        handler.restructures[2],
    )
    handler.convert_model_to_meta()
    source = handler.mapping[model.block]
    original_norm = source.norm
    handler.apply_restructuring()
    target = model.block
    assert handler.restructured_modules == {target}
    assert isinstance(target[1], NormCalibrationModule)
    modules = dict(target.named_modules(prefix="block"))
    handler.configure_onload(torch.device("cpu"))
    handler.onload_and_restructure_modules(modules, torch.device("cpu"))
    assert original_norm._parameters  # The source subtree release must retain it.
    with torch.no_grad():
        target[1].weight.add_(2)
    handler.restore_restructured_modules(modules, offload_sources=False)
    assert model.block[1] is original_norm
    assert modules["block.1"] is original_norm
    torch.testing.assert_close(original_norm.weight, torch.full((16,), 2.0))
    handler.finish()
