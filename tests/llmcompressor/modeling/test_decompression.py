import copy

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
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modeling.decompression import DecompressionRestructure
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
    assert not list(DecompressionRestructure().targets(model))
    scheme = layer.quantization_scheme
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
    DecompressionRestructure().restructure(reference, None)
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
