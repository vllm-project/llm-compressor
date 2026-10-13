import pytest
import torch
from torch.fx import Node
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import (
    _resolve_sequential_targets,
    trace_subgraphs,
)


@pytest.fixture
def tiny_llama():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        _attn_implementation="sdpa",
    )
    return LlamaForCausalLM(config).eval()


def test_t5_attention_modules_use_shared_attention_detector():
    config = T5Config(
        vocab_size=64,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
    )
    model = T5ForConditionalGeneration(config)
    targets = _resolve_sequential_targets(model, ["Linear"])

    target_names = {name for name, module in model.named_modules() if module in targets}
    assert "encoder.block.0.layer.0.SelfAttention" in target_names
    assert "decoder.block.0.layer.0.SelfAttention" in target_names
    assert "decoder.block.0.layer.1.EncDecAttention" in target_names


def _trace_and_run_attention(model, sample_input):
    subgraphs = trace_subgraphs(
        model,
        sample_input,
        sequential_targets=["Linear"],
        ignore=DatasetArguments().tracing_ignore,
    )

    attention_name = next(
        name
        for name, module in model.named_modules()
        if module is model.model.layers[0].self_attn
    )
    attention_subgraph = next(
        subgraph
        for subgraph in subgraphs
        if any(
            node.op == "call_module" and node.target == attention_name
            for node in subgraph.graph.nodes
        )
    )
    attention_call = next(
        node
        for node in attention_subgraph.graph.nodes
        if node.op == "call_module" and node.target == attention_name
    )
    mask_node = attention_call.kwargs["attention_mask"]
    assert isinstance(mask_node, Node)

    preamble_outputs = subgraphs[0].forward(model, **sample_input)
    mask = preamble_outputs[mask_node.name]
    attention_outputs = attention_subgraph.forward(
        model,
        **{name: preamble_outputs[name] for name in attention_subgraph.input_names},
    )
    return mask, attention_outputs


@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        ([[1, 2, 3, 4]], [[1, 1, 1, 1]]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
    ],
    ids=["batch1", "batch2-truncate"],
)
def test_unpadded_batches_do_not_materialize_quadratic_attention_mask(
    tiny_llama, input_ids, attention_mask
):
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    mask, attention_outputs = _trace_and_run_attention(
        tiny_llama,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )

    assert mask is None
    assert any(
        isinstance(output, torch.Tensor) and output.shape[:2] == input_ids.shape
        for output in attention_outputs.values()
    )


def test_padded_batch_materializes_attention_mask(tiny_llama):
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    mask, attention_outputs = _trace_and_run_attention(
        tiny_llama,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (2, 1, 4, 4)
    assert any(
        isinstance(output, torch.Tensor) and output.shape[:2] == input_ids.shape
        for output in attention_outputs.values()
    )
