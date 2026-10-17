# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Random official Transformers GLM model; never loads an external checkpoint."""

import json
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modeling.moe.linearize import get_linearized_moes, repack_moe
from llmcompressor.modifiers.transform import QuaRotModifier


@pytest.mark.parametrize("indexers", [["full", "shared"], ["full", "full"]])
@torch.no_grad()
def test_official_glm_oneshot_and_local_reload(tmp_path, monkeypatch, indexers):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = GlmMoeDsaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        index_topk=2,
        index_head_dim=64,
        index_n_heads=2,
        indexer_types=indexers,
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).eval()
    # Oneshot validates the original config even for an in-memory model.
    config.save_pretrained(tmp_path / "input-config")
    model.config._name_or_path = str(tmp_path / "input-config")
    for module in model.modules():
        if module.__class__.__name__.endswith("RMSNorm"):
            module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    tokens = torch.tensor([[1, 7, 4, 5, 9], [9, 3, 5, 1, 7]])
    expected = model(tokens).logits
    # A generated tokenizer prevents even an attempted auto-lookup in pre_process.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=[QuaRotModifier(block_size=32)],
        pipeline="datafree",
    )
    assert len(get_linearized_moes(model)) == 1
    actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert model.config.quarot_config["status"] == "applied"

    # Save only this tiny generated fixture, repacked to the native HF layout.
    repack_moe(model)
    model.save_pretrained(tmp_path, save_compressed=False)
    restored = GlmMoeDsaForCausalLM.from_pretrained(
        tmp_path, local_files_only=True, attn_implementation="eager"
    ).eval()
    torch.testing.assert_close(restored(tokens).logits, actual, atol=2e-6, rtol=2e-5)
    assert restored.config.quarot_config == model.config.quarot_config
    with pytest.raises(ValueError, match="already has QuaRot"):
        QuaRotModifier().initialize(State(model=restored))
    if directory := os.environ.get("QUAROT_REPORT_DIR"):
        report = {
            "model_class": type(model).__name__,
            "fixture_seed": 42,
            "config": config.to_dict(),
            "logits_max_abs_error": (actual - expected).abs().max().item(),
            "reload_logits_max_abs_error": (restored(tokens).logits - actual)
            .abs()
            .max()
            .item(),
            "atol": 2e-6,
            "rtol": 2e-5,
            "real_weights": False,
        }
        Path(directory, f"l3-official-glm-{'-'.join(indexers)}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
