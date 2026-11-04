"""
Test that GPTQ quantizes MoE experts which receive no calibration tokens.

With `moe_calibrate_all_experts=False`, an expert the router never picks is never
called, so its GPTQ hook never fires. It must still be quantized from its own
weights rather than saved with a zero scale or zeroed weights.
"""

import io

import torch
from datasets import Dataset
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3MoeConfig,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

NUM_EXPERTS = 8


def _make_tiny_moe(path) -> tuple[torch.nn.Module, PreTrainedTokenizerFast]:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=1,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    vocab = {"[UNK]": 0, **{str(index): index for index in range(1, 128)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return model, tokenizer


def test_gptq_quantizes_unrouted_experts(tmp_path):
    model, tokenizer = _make_tiny_moe(tmp_path / "model")

    # one 4-token sample with top-1 routing leaves at least 4 experts unrouted
    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
    )
    logs = io.StringIO()
    handler = logger.add(logs, level="WARNING")
    try:
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=dataset,
            num_calibration_samples=1,
            max_seq_length=4,
            moe_calibrate_all_experts=False,
            recipe=GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head", "re:.*mlp.gate$"],
            ),
        )
    finally:
        logger.remove(handler)

    experts = model.model.layers[0].mlp.experts
    for index in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(experts[index], name)
            assert torch.all(linear.weight_scale > 0), f"expert {index} {name} scale"
            assert linear.weight.abs().sum() > 0, f"expert {index} {name} zeroed"
    assert "received no calibration tokens" in logs.getvalue()
