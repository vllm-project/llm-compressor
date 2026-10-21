import pytest
from datasets import Dataset
from loguru import logger
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.core import Event, EventType
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from tests.llmcompressor.modifiers.conf import LifecyleTestingHarness

IGNORED_MODULE = "lm_head"
CONTROL_MODULE = "model.layers.0.mlp.down_proj"
SPARSE_MODULE = "model.layers.0.mlp.up_proj"

MODIFIERS = [SparseGPTModifier, WandaPruningModifier]


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


def _targeted_modules(modifier):
    """
    Run a modifier up to the start of calibration and return the model together
    with the mapping of the modules that the modifier decided to sparsify.
    """
    harness = LifecyleTestingHarness(model=_build_model(), start=-1)
    modifier.initialize(harness.get_state())
    modifier.update_event(harness.get_state(), Event(type_=EventType.CALIBRATION_START))

    return harness.get_state().model, modifier._module_names


def _zero_fraction(model, name):
    return (model.get_submodule(name).weight == 0).float().mean().item()


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
@pytest.mark.parametrize(
    "ignore",
    [
        ["lm_head"],
        ["re:.*lm_head$"],
        ["re:.*lm_head"],
    ],
)
def test_ignore_matches_lm_head(modifier_class, ignore):
    # The default `targets` are `["Linear"]`, which makes every target layer the
    # prunable module itself. The module name used to be built by appending the
    # relative name unconditionally, producing "lm_head." instead of "lm_head".
    # Exact names and patterns anchored with "$" can never match that.
    modifier = modifier_class(sparsity=0.5, targets=["Linear"], ignore=ignore)
    model, module_names = _targeted_modules(modifier)

    assert model.get_submodule(IGNORED_MODULE) not in module_names
    assert model.get_submodule(CONTROL_MODULE) in module_names


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_ignore_matches_nested_module_by_exact_name(modifier_class):
    modifier = modifier_class(sparsity=0.5, targets=["Linear"], ignore=[CONTROL_MODULE])
    model, module_names = _targeted_modules(modifier)

    assert model.get_submodule(CONTROL_MODULE) not in module_names
    assert model.get_submodule(IGNORED_MODULE) in module_names


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_targeted_module_names_are_fully_qualified(modifier_class):
    modifier = modifier_class(sparsity=0.5, targets=["Linear"])
    _, module_names = _targeted_modules(modifier)

    names = set(module_names.values())
    assert IGNORED_MODULE in names
    assert CONTROL_MODULE in names
    assert all(not name.endswith(".") for name in names)


@pytest.mark.unit
@pytest.mark.parametrize("modifier_class", MODIFIERS)
def test_ignored_module_is_not_hooked(modifier_class):
    modifier = modifier_class(sparsity=0.5, targets=["Linear"], ignore=["lm_head"])
    model, _ = _targeted_modules(modifier)

    assert len(model.get_submodule(IGNORED_MODULE)._forward_hooks) == 0
    assert len(model.get_submodule(CONTROL_MODULE)._forward_hooks) == 1


@pytest.mark.unit
def test_lm_head_warning_is_emitted_when_not_ignored():
    # The warning about sparsifying `lm_head` used the same mangled name and
    # therefore never fired.
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        modifier = SparseGPTModifier(sparsity=0.5, targets=["Linear"])
        _targeted_modules(modifier)
    finally:
        logger.remove(handler_id)

    assert any("lm_head" in message for message in logs)


@pytest.mark.integration
@pytest.mark.parametrize("modifier_class", MODIFIERS)
@pytest.mark.parametrize(
    "ignore,ignored_module",
    [
        (["lm_head"], "lm_head"),
        (["re:.*lm_head$"], "lm_head"),
        (["re:.*lm_head"], "lm_head"),
        (["model.layers.0.mlp.down_proj"], "model.layers.0.mlp.down_proj"),
    ],
)
def test_oneshot_does_not_prune_ignored_modules(
    modifier_class, ignore, ignored_module, tmp_path
):
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
        recipe=[modifier_class(sparsity=0.5, ignore=ignore)],
        num_calibration_samples=4,
        pipeline="sequential",
    )

    assert _zero_fraction(model, ignored_module) == 0.0
    assert _zero_fraction(model, SPARSE_MODULE) == pytest.approx(0.5, abs=0.05)
