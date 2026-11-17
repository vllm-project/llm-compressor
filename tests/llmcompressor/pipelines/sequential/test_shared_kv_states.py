import tempfile
import warnings
from collections import UserDict

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import (
    find_modules_outside_subgraphs,
    trace_subgraphs,
)


def _tiny_gemma4_with_kv_sharing() -> Gemma4ForCausalLM:
    # the last two layers reuse the kv states of the last sliding and full layers
    config = Gemma4TextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    torch.manual_seed(0)
    return Gemma4ForCausalLM(config).eval()


def test_shared_kv_states_cache_keeps_user_dict_by_reference():
    shared = UserDict()
    cache = IntermediatesCache.empty(1, torch.device("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache.update(0, {"shared": shared})

    fetched = cache.fetch(0, ["shared"])["shared"]
    assert fetched is shared

    # writes made by one subgraph are seen by the next one
    fetched["full_attention"] = torch.ones(1)
    assert "full_attention" in cache.fetch(0, ["shared"])["shared"]


@torch.no_grad()
def test_sequential_subgraphs_share_kv_states():
    model = _tiny_gemma4_with_kv_sharing()
    batches = [
        {
            "input_ids": torch.randint(0, 128, (1, 12), generator=g),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }
        for g in (torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    ]

    subgraphs = trace_subgraphs(
        model,
        batches[0],
        sequential_targets=["Gemma4TextDecoderLayer"],
        ignore=DatasetArguments().tracing_ignore,
    )

    # run the subgraphs the way the sequential pipeline does: one subgraph at a
    # time over every batch, passing values between subgraphs through the cache
    cache = IntermediatesCache.empty(len(batches), torch.device("cpu"))
    for batch_idx, batch in enumerate(batches):
        cache.update(batch_idx, batch)

    outputs = [None] * len(batches)
    for subgraph_idx, subgraph in enumerate(subgraphs):
        for batch_idx in range(len(batches)):
            inputs = cache.fetch(batch_idx, subgraph.input_names)
            output = subgraph.forward(model, **inputs)
            if subgraph_idx < len(subgraphs) - 1:
                cache.update(batch_idx, output)
                cache.delete(batch_idx, subgraph.consumed_names)
            else:
                outputs[batch_idx] = output

    for batch, output in zip(batches, outputs):
        expected = model(**batch, use_cache=False).logits
        assert torch.allclose(output["logits"], expected, atol=1e-5)


def _tiny_gemma4_conditional_with_kv_sharing() -> Gemma4ForConditionalGeneration:
    text_config = Gemma4TextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        sliding_window=8,
        max_position_embeddings=64,
        attention_k_eq_v=True,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text_config, vision_config=vision_config)
    ).eval()


def _tiny_gemma4_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"t{i}": i for i in range(128)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="t0", unk_token="t0"
    )


@torch.no_grad()
def test_conditional_generation_subgraphs_include_per_layer_projection():
    model = _tiny_gemma4_conditional_with_kv_sharing()
    batches = [
        {
            "input_ids": torch.randint(0, 128, (1, 12), generator=g),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }
        for g in (torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    ]

    subgraphs = trace_subgraphs(
        model,
        batches[0],
        sequential_targets=["Gemma4TextDecoderLayer"],
        ignore=DatasetArguments().tracing_ignore,
    )

    preamble_modules = set(subgraphs[0].submodules(model))
    outside = find_modules_outside_subgraphs(model, subgraphs)
    lm = model.model.language_model
    for attr_name in (
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    ):
        qualified = f"model.language_model.{attr_name}"
        assert getattr(lm, attr_name) in preamble_modules, qualified
        assert qualified not in outside

    cache = IntermediatesCache.empty(len(batches), torch.device("cpu"))
    for batch_idx, batch in enumerate(batches):
        cache.update(batch_idx, batch)

    outputs = [None] * len(batches)
    for subgraph_idx, subgraph in enumerate(subgraphs):
        for batch_idx in range(len(batches)):
            inputs = cache.fetch(batch_idx, subgraph.input_names)
            output = subgraph.forward(model, **inputs)
            if subgraph_idx < len(subgraphs) - 1:
                cache.update(batch_idx, output)
                cache.delete(batch_idx, subgraph.consumed_names)
            else:
                outputs[batch_idx] = output

    for batch, output in zip(batches, outputs):
        expected = model(**batch, use_cache=False).logits
        assert torch.allclose(output["logits"], expected, atol=1e-5)


def test_gemma4_conditional_gptq_oneshot_cpu():
    tokenizer = _tiny_gemma4_tokenizer()
    dataset = Dataset.from_list(
        [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "attention_mask": [1] * 8},
            {"input_ids": [2, 3, 4, 5, 6, 7, 8, 9], "attention_mask": [1] * 8},
        ]
    )
    ignore = ["lm_head", "re:.*vision.*", "re:.*audio.*"]
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 16,
            },
        }
    }

    with tempfile.TemporaryDirectory() as tmp:
        gptq_model = _tiny_gemma4_conditional_with_kv_sharing()
        gptq_model.config.save_pretrained(tmp)
        gptq_model.config._name_or_path = tmp
        oneshot(
            model=gptq_model,
            processor=tokenizer,
            dataset=dataset,
            recipe=[
                GPTQModifier(
                    config_groups=config_groups,
                    ignore=ignore,
                    block_size=16,
                )
            ],
            num_calibration_samples=2,
        )
        assert hasattr(
            gptq_model.model.language_model.per_layer_model_projection,
            "weight_scale",
        )
        assert hasattr(
            gptq_model.model.language_model.layers[4].self_attn.q_proj,
            "weight_scale",
        )


def test_gemma4_conditional_awq_oneshot_cpu(monkeypatch):
    if not torch.accelerator.is_available():
        # IntermediatesCache.pin_memory calls torch.Tensor.pin_memory(), which
        # raises RuntimeError when no accelerator/CUDA GPU is visible.
        monkeypatch.setattr(
            IntermediatesCache, "pin_memory", lambda self, *args, **kwargs: None
        )

    tokenizer = _tiny_gemma4_tokenizer()
    dataset = Dataset.from_list(
        [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "attention_mask": [1] * 8},
            {"input_ids": [2, 3, 4, 5, 6, 7, 8, 9], "attention_mask": [1] * 8},
        ]
    )
    ignore = ["lm_head", "re:.*vision.*", "re:.*audio.*"]
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": False,
                "strategy": "group",
                "group_size": 16,
            },
        }
    }

    with tempfile.TemporaryDirectory() as tmp:
        awq_model = _tiny_gemma4_conditional_with_kv_sharing()
        awq_model.config.save_pretrained(tmp)
        awq_model.config._name_or_path = tmp
        oneshot(
            model=awq_model,
            processor=tokenizer,
            dataset=dataset,
            recipe=[
                AWQModifier(),
                QuantizationModifier(
                    config_groups=config_groups,
                    ignore=ignore,
                ),
            ],
            num_calibration_samples=2,
        )
        lm = awq_model.model.language_model
        assert hasattr(lm.layers[0].self_attn.k_proj, "weight_scale")
        assert hasattr(lm.layers[4].self_attn.q_proj, "weight_scale")
        assert hasattr(lm.layers[5].self_attn.q_proj, "weight_scale")
        assert hasattr(lm.layers[0].mlp.down_proj, "weight_scale")
        assert not hasattr(
            awq_model.model.vision_tower.encoder.layers[0].self_attn.q_proj,
            "weight_scale",
        )
