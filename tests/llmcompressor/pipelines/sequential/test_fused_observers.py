import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.core import LifecycleCallbacks
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential import helpers


@pytest.fixture
def calibration_model(tmp_path):
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
    data = DataLoader(
        [{"input_ids": torch.randint(0, 32, (1, 8))} for _ in range(4)],
        batch_size=None,
    )
    return model, data


@pytest.mark.parametrize("modifier_cls", [QuantizationModifier, GPTQModifier])
@pytest.mark.parametrize("targets_per_subgraph", [1, "auto"])
def test_split_fused_group_raises_before_calibration(
    calibration_model, monkeypatch, modifier_cls, targets_per_subgraph
):
    model, data = calibration_model
    # Exercise the auto planner's fallback to one target per subgraph too.
    monkeypatch.setattr(helpers, "plan_subgraph_memory", lambda *args: None)
    epochs = []
    monkeypatch.setattr(LifecycleCallbacks, "sequential_epoch_end", epochs.append)

    with pytest.raises(
        ValueError,
        match="gate_proj.*same subgraph.*sequential_targets_per_subgraph",
    ):
        oneshot(
            model=model,
            processor=object(),
            dataset=data,
            recipe=modifier_cls(
                targets="Linear", scheme="NVFP4A16", ignore=["lm_head"]
            ),
            pipeline="sequential",
            sequential_targets=["Linear"],
            sequential_targets_per_subgraph=targets_per_subgraph,
            stage_weights_in_pinned_memory=False,
        )

    assert epochs == []
    assert model.model.layers[0].mlp.gate_proj.weight.is_meta
    assert model.model.layers[0].mlp.up_proj.weight.is_meta


@pytest.mark.parametrize("modifier_cls", [QuantizationModifier, GPTQModifier])
@pytest.mark.parametrize(
    "weight_observer,layerwise_compression",
    [("memoryless_minmax", False), ("imatrix_mse", True)],
)
@pytest.mark.parametrize(
    "scheme,pipeline,targets,targets_per_subgraph",
    [
        ("NVFP4A16", "sequential", ["LlamaDecoderLayer"], 1),
        ("NVFP4A16", "sequential", ["Linear"], 10),
        ("NVFP4A16", "basic", ["Linear"], 1),
        ("W8A8", "sequential", ["Linear"], 1),
    ],
    ids=["common-parent", "grouped-targets", "whole-model", "unfused"],
)
def test_supported_fusion_boundaries(
    calibration_model,
    modifier_cls,
    weight_observer,
    layerwise_compression,
    scheme,
    pipeline,
    targets,
    targets_per_subgraph,
):
    model, data = calibration_model
    oneshot(
        model=model,
        processor=object(),
        dataset=data,
        recipe=modifier_cls(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
            weight_observer=weight_observer,
        ),
        pipeline=pipeline,
        sequential_targets=targets,
        sequential_targets_per_subgraph=targets_per_subgraph,
        stage_weights_in_pinned_memory=False,
        layerwise_compression=layerwise_compression,
    )
    assert all(not tensor.is_meta for tensor in model.state_dict().values())
    if scheme == "NVFP4A16":
        layer = model.model.layers[0]
        for group in (
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        ):
            for module in group[1:]:
                torch.testing.assert_close(
                    group[0].weight_global_scale, module.weight_global_scale
                )
