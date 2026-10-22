from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import OffloadCache
from compressed_tensors.offload.module import offload_module
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.transform.factory.base import TransformBase
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import EventType, State, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from llmcompressor.pipelines.sequential import SequentialPipeline


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
@pytest.mark.parametrize("modifier_cls", [QuIPModifier, SpinQuantModifier])
@pytest.mark.parametrize(
    "onload_device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_initialization_tracks_and_offloads_transform_children(
    modifier_cls, offload_device, onload_device, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "llmcompressor.pipelines.sequential.pipeline.get_main_device",
        lambda: torch.device(onload_device),
    )
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            tie_word_embeddings=False,
        )
    ).eval()
    batches = [{"input_ids": torch.randint(0, 64, (1, 4))}]
    kwargs = {"rotations": ["R3", "R4"]} if modifier_cls is SpinQuantModifier else {}
    # Apply the same transforms eagerly to check that scoped placement preserves
    # their values, including the paired attention transforms.
    eager = deepcopy(model)
    eager_modifier = modifier_cls(**kwargs)
    eager_state = State(model=eager)
    eager_modifier.initialize(eager_state)
    eager_modifier.on_sequential_epoch_initialize(eager_state, None)
    eager_modifier.on_calibration_start(eager_state, None)
    with torch.no_grad():
        expected = eager(**batches[0]).logits

    offload_args = {"onload_device": onload_device, "offload_device": offload_device}
    if offload_device == "disk":
        offload_args["offload_dir"] = str(tmp_path)
    for module in model.modules():
        offload_module(module, **offload_args)

    modifier = modifier_cls(**kwargs)
    with create_session() as session:
        session.initialize(model=model, recipe=modifier, calib_data=batches)
        SequentialPipeline()(
            model,
            batches,
            DatasetArguments(
                sequential_targets=["LlamaDecoderLayer"],
                layerwise_decompression=False,
            ),
        )
        session.finalize()

    transforms = [
        module for module in model.modules() if isinstance(module, TransformBase)
    ]
    assert transforms
    for transform in transforms:
        assert isinstance(transform._parameters, OffloadCache)
        assert str(transform._parameters.offload_device) == offload_device
        with OffloadCache.disable_onloading():
            assert transform.weight.device.type == (
                "meta" if offload_device == "disk" else "cpu"
            )
    assert not modifier._transform_factories
    if modifier_cls is SpinQuantModifier:
        for layer in model.model.layers:
            assert isinstance(layer.self_attn.impl._parameters, OffloadCache)
            assert isinstance(layer.self_attn.kv_cache._parameters, OffloadCache)
    with torch.no_grad():
        torch.testing.assert_close(model(**batches[0]).logits.cpu(), expected)


@pytest.mark.parametrize("modifier_cls", [QuantizationModifier, GPTQModifier])
def test_scope_cleanup_preserves_persistent_calibration(modifier_cls):
    torch.manual_seed(13)
    model = torch.nn.Sequential(*(torch.nn.Linear(8, 8) for _ in range(3)))
    persistent = model[0]
    modifier = modifier_cls(
        config_groups={
            "static": {
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "strategy": "channel"},
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "dynamic": False,
                },
            }
        }
    )
    with create_session() as session:
        session.initialize(model=model, recipe=modifier)
        event = session.lifecycle.event
        event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[persistent])
        assert not hasattr(persistent, "weight_observer")
        event(EventType.CALIBRATION_START, modules=[persistent])
        observer = persistent.input_observer

        # A scope with no targets must not finish persistent calibration or fail GPTQ.
        event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[])
        event(EventType.CALIBRATION_START, modules=[])
        event(EventType.SEQUENTIAL_EPOCH_END, modules=[])
        event(EventType.CALIBRATION_END, modules=[])
        assert not modifier.ended_

        for layer in (model[1], model[2], model[1]):
            event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[layer])
            assert not hasattr(layer, "weight_observer")
            assert not layer._forward_hooks
            event(EventType.CALIBRATION_START, modules=[layer])
            with torch.no_grad():
                layer(persistent(torch.randn(16, 8)))
            event(EventType.SEQUENTIAL_EPOCH_END, modules=[layer])
            event(EventType.CALIBRATION_END, modules=[layer])

            assert layer.quantization_status == QuantizationStatus.FROZEN
            assert not hasattr(layer, "weight_observer")
            assert not layer._forward_hooks and not layer._forward_pre_hooks
            assert persistent.quantization_status == QuantizationStatus.CALIBRATION
            assert persistent.input_observer is observer
            assert observer.has_statistics
            assert persistent._forward_pre_hooks
            assert not modifier.ended_
            if isinstance(modifier, GPTQModifier):
                assert persistent in modifier._hessians
                assert layer not in modifier._hessians

        event(EventType.SEQUENTIAL_EPOCH_END, modules=[persistent])
        event(EventType.CALIBRATION_END, modules=[persistent])
        assert modifier.ended_
        assert not modifier._hooks
        assert not modifier._hook_owners
        session.finalize()


@pytest.mark.parametrize("pipeline_name", ["basic", "datafree"])
def test_nonsequential_pipelines_initialize_before_calibration(pipeline_name):
    from llmcompressor.pipelines import CalibrationPipeline

    model = torch.nn.Linear(128, 8)
    modifier = QuantizationModifier(scheme="W8A16")
    with create_session() as session:
        session.initialize(model=model, recipe=modifier)
        assert not hasattr(model, "quantization_scheme")
        pipeline = CalibrationPipeline.from_modifiers([modifier], user=pipeline_name)
        pipeline(model, [{"input": torch.randn(2, 128)}], DatasetArguments())
        assert model.quantization_status == QuantizationStatus.FROZEN
        assert torch.isfinite(model.weight_scale).all()
        assert not hasattr(model, "weight_observer")
        assert not modifier._hooks
        session.finalize()
