import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.sparsegpt.sgpt_sparsify import sparsify_weight
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier

MODIFIERS = [SparseGPTModifier, WandaPruningModifier]

SPARSE_MODULE = "model.layers.0.mlp.up_proj"


def _build_config():
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=False,
    )


def _build_model(config=None):
    return LlamaForCausalLM(config or _build_config())


def _build_tokenizer(directory):
    # a tiny local tokenizer, so that the test needs no network access
    vocab = {f"tok{index}": index for index in range(60)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="tok0", pad_token="tok0"
    )
    fast_tokenizer.save_pretrained(directory)

    return fast_tokenizer


def _sparse_gpt(sparsity):
    module = torch.nn.Linear(64, 32)
    _, weight = sparsify_weight(
        module=module,
        hessians_dict={module: torch.eye(64)},
        sparsity=sparsity,
        prune_n=0,
        prune_m=0,
        block_size=128,
        dampening_frac=0.01,
        preserve_sparsity_mask=False,
    )
    return (weight == 0).float().mean().item()


@pytest.mark.unit
def test_sparse_gpt_prunes_every_weight_for_a_sparsity_of_one():
    # the threshold index used to be `tmp.numel()` for a sparsity of 1.0, which
    # is one past the end of the sorted values
    assert _sparse_gpt(1.0) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
@pytest.mark.parametrize("sparsity", [-0.1, 1.5, 2.0])
def test_out_of_range_sparsity_is_rejected(modifier_class, sparsity):
    # an out of range sparsity used to be silently misinterpreted, a negative
    # index wrapped around and pruned about 90% of the weights
    with pytest.raises(ValueError, match="between 0 and 1"):
        modifier_class(sparsity=sparsity)


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
@pytest.mark.parametrize("sparsity", [True, False, [True, 0.5]])
def test_boolean_sparsity_is_rejected(modifier_class, sparsity):
    # `bool` is a subclass of `int`, so `sparsity=True` used to be accepted and
    # converted to 1.0, which prunes every weight
    with pytest.raises(ValueError, match="must be a number"):
        modifier_class(sparsity=sparsity)


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_in_range_sparsity_is_accepted(modifier_class):
    assert modifier_class(sparsity=0.0) is not None
    assert modifier_class(sparsity=0.5) is not None
    assert modifier_class(sparsity=1.0) is not None
    assert modifier_class(sparsity=[0.5, 0.25]) is not None


@pytest.mark.integration
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_oneshot_prunes_every_weight_for_a_sparsity_of_one(modifier_class, tmp_path):
    config = _build_config()
    source = tmp_path / "source"
    config.save_pretrained(source)
    _build_tokenizer(source)

    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 8, "attention_mask": [[1, 1, 1, 1]] * 8}
    )

    model = _build_model(config)
    model.config.name_or_path = str(source)
    oneshot(
        model=model,
        dataset=dataset,
        recipe=[modifier_class(sparsity=1.0)],
        num_calibration_samples=4,
        pipeline="sequential",
    )

    zero_fraction = (
        (model.get_submodule(SPARSE_MODULE).weight == 0).float().mean().item()
    )
    assert zero_fraction == 1.0
