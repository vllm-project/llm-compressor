import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
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


def _target_layers(modifier, model):
    state = State()
    state.update(model=model, device="cpu", calib_data=None)
    modifier.initialize(state)

    return modifier._target_layers


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_string_targets_match_the_same_layers_as_a_list(modifier_class):
    # `match_named_modules` iterates over its targets, so a plain string used
    # to be iterated character by character and match nothing at all
    model = _build_model()

    from_list = _target_layers(modifier_class(sparsity=0.5, targets=["Linear"]), model)
    from_string = _target_layers(modifier_class(sparsity=0.5, targets="Linear"), model)

    assert from_list
    assert set(from_string) == set(from_list)


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_special_target_constants_are_expanded(modifier_class):
    model = _build_model()

    target_layers = _target_layers(
        modifier_class(sparsity=0.5, targets="__ALL_PRUNABLE__"), model
    )

    assert target_layers


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_unsupported_all_target_raises(modifier_class):
    # `__ALL__` is no longer supported, but it used to be silently ignored
    with pytest.raises(ValueError, match="__ALL__"):
        modifier_class(sparsity=0.5, targets="__ALL__")


@pytest.mark.integration
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_oneshot_prunes_with_string_targets(modifier_class, tmp_path):
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
        recipe=[modifier_class(sparsity=0.5, targets="Linear")],
        num_calibration_samples=4,
        pipeline="sequential",
    )

    zero_fraction = (
        (model.get_submodule(SPARSE_MODULE).weight == 0).float().mean().item()
    )
    assert zero_fraction == pytest.approx(0.5, abs=0.05)
