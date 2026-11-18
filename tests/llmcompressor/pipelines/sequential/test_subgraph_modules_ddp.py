"""
Distributed modifiers such as GPTQ split the modules of a subgraph across ranks
by iterating `Subgraph.submodule_dict`, so every rank must see the same names in
the same order. Each torchrun rank is its own process with its own string hash
seed, so an order that depended on set or hash iteration would differ here.
"""

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from tests.testing_utils import torchrun


@pytest.mark.unit
@torchrun(world_size=2, init_dist=False)
def test_subgraph_module_order_matches_across_ranks():
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(config)
        sample_input = {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        subgraphs = trace_subgraphs(
            model,
            sample_input,
            sequential_targets=["LlamaDecoderLayer"],
            ignore=DatasetArguments().tracing_ignore,
        )

        local = [list(subgraph.submodule_dict(model)) for subgraph in subgraphs]
        assert any(local), "expected subgraphs to own modules"

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        assert all(names == gathered[0] for names in gathered)
    finally:
        dist.destroy_process_group()
