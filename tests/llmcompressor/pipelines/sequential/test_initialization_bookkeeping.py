from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import OffloadCache
from compressed_tensors.offload.module import offload_module
from compressed_tensors.transform.factory.base import TransformBase
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import State, create_session
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
