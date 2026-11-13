import warnings
from collections import UserDict

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs


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
