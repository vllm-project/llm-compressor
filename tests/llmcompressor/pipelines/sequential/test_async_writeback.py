# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.offload import AsyncWriter, offload_module
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    LlamaConfig,
    Qwen3MoeConfig,
)

from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers.gptq import GPTQModifier


@pytest.mark.parametrize("config_cls", [LlamaConfig, Gemma2Config, Qwen3MoeConfig])
@pytest.mark.parametrize("disk_offload", [False, True])
@pytest.mark.parametrize("streaming_save", [False, True])
def test_restructured_checkpoint_matches_calibrated_model(
    tmp_path, monkeypatch, config_cls, disk_offload, streaming_save
):
    """Async writes retain calibrated experts, restored norms, and all saved keys."""
    torch.manual_seed(42)
    config = config_cls(
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
        tie_word_embeddings=False,
    )
    source_dir = tmp_path / "source"
    config.save_pretrained(source_dir)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.config.name_or_path = str(source_dir)
    if disk_offload:
        for module in list(model.modules()):
            if not module._parameters:
                continue
            offload_module(
                module,
                onload_device="cpu",
                offload_device="disk",
                offload_dir=tmp_path / "offload",
            )

    submitted = []
    submit = AsyncWriter.submit

    def track_submit(self, path, tensors):
        submitted.append(path)
        return submit(self, path, tensors)

    monkeypatch.setattr(AsyncWriter, "submit", track_submit)
    output_dir = tmp_path / "checkpoint"
    run = Oneshot(
        model=model,
        processor=object(),
        recipe=GPTQModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                    },
                }
            },
            ignore=["lm_head"],
        ),
        pipeline="sequential",
        layerwise_compression=True,
        streaming_save=streaming_save,
        output_dir=str(output_dir),
    )
    data = DataLoader([{"input_ids": torch.tensor([[1, 2, 3, 4]])}], batch_size=None)
    try:
        run.apply_recipe_modifiers(data)
        expected = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        # Disk caches have moved to retained source holders before the pipeline
        # checks for them. Ordinary disk offload must still submit async writes.
        if not streaming_save:
            assert bool(submitted) is disk_offload
        model.save_pretrained(output_dir, _async_writer=run.async_writer)

        index_path = output_dir / "model.safetensors.index.json"
        if streaming_save:
            index = json.loads(index_path.read_text())
            assert set(index["weight_map"]) == set(expected)
        saved = {}
        for shard in output_dir.glob("*.safetensors"):
            saved.update(load_file(shard))
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
        assert (output_dir / "config.json").is_file()
    finally:
        if run.async_writer is not None:
            run.async_writer.abort()
