import pytest
from loguru import logger
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.core import Event, EventType
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from tests.llmcompressor.modifiers.conf import LifecyleTestingHarness

IGNORED_MODULE = "lm_head"
CONTROL_MODULE = "model.layers.0.mlp.down_proj"

MODIFIERS = [SparseGPTModifier, WandaPruningModifier]


def _build_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _start_calibration(modifier):
    """
    Initialize the modifier on a tiny model and start calibration, returning the
    model and the names of the modules the modifier will prune
    """
    harness = LifecyleTestingHarness(model=_build_model(), start=-1)
    state = harness.get_state()
    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    return state.model, modifier._module_names


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", MODIFIERS)
@pytest.mark.parametrize("ignore", [["lm_head"], ["re:.*lm_head$"], ["re:.*lm_head"]])
def test_ignore_matches_lm_head(modifier_cls, ignore):
    # with the default targets, each target layer is a single Linear module, so
    # the module name must not get a trailing "." from an empty relative name
    model, module_names = _start_calibration(
        modifier_cls(sparsity=0.5, targets=["Linear"], ignore=ignore)
    )

    assert model.get_submodule(IGNORED_MODULE) not in module_names
    assert model.get_submodule(CONTROL_MODULE) in module_names
    assert len(model.get_submodule(IGNORED_MODULE)._forward_hooks) == 0
    assert len(model.get_submodule(CONTROL_MODULE)._forward_hooks) == 1


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", MODIFIERS)
def test_ignore_matches_nested_module_by_exact_name(modifier_cls):
    model, module_names = _start_calibration(
        modifier_cls(sparsity=0.5, targets=["Linear"], ignore=[CONTROL_MODULE])
    )

    assert model.get_submodule(CONTROL_MODULE) not in module_names
    assert model.get_submodule(IGNORED_MODULE) in module_names


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", MODIFIERS)
def test_module_names_are_fully_qualified(modifier_cls):
    _, module_names = _start_calibration(modifier_cls(sparsity=0.5, targets=["Linear"]))

    names = set(module_names.values())
    assert {IGNORED_MODULE, CONTROL_MODULE} <= names
    assert not any(name.endswith(".") for name in names)


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", MODIFIERS)
def test_lm_head_warning_when_not_ignored(modifier_cls):
    messages = []
    handler_id = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        _start_calibration(modifier_cls(sparsity=0.5, targets=["Linear"]))
    finally:
        logger.remove(handler_id)

    assert any("lm_head" in message for message in messages)
