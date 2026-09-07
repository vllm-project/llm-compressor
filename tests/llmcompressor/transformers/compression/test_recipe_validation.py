import pytest
import torch
import torch.nn as nn
from accelerate.accelerator import get_state_dict_offloaded_model
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor.core import active_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.transformers.compression.recipe_validation import (
    get_unmatched_ignore_entries,
    warn_on_unmatched_ignore_entries,
)

TINY_MODEL = "nm-testing/tinysmokellama-3.2"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _modifier(ignore):
    return QuantizationModifier(targets=["Linear"], scheme="W8A8", ignore=ignore)


@pytest.fixture
def warning_logs():
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    yield logs
    logger.remove(handler_id)


@pytest.mark.unit
def test_unmatched_ignore_entry_is_reported():
    model = DummyModel()
    modifier = _modifier(["nonexistent_module"])

    assert get_unmatched_ignore_entries(model, [modifier]) == [
        ("QuantizationModifier", "nonexistent_module")
    ]


@pytest.mark.unit
def test_matching_ignore_entries_are_not_reported():
    model = DummyModel()
    # by name, by regex, and by module class
    modifier = _modifier(["layer1", "re:layer.*", "LayerNorm"])

    assert get_unmatched_ignore_entries(model, [modifier]) == []


@pytest.mark.unit
def test_only_the_unmatched_entries_are_reported():
    model = DummyModel()
    modifier = _modifier(["layer1", "nonexistent_module", "norm"])

    assert get_unmatched_ignore_entries(model, [modifier]) == [
        ("QuantizationModifier", "nonexistent_module")
    ]


@pytest.mark.unit
def test_entries_are_reported_per_modifier():
    model = DummyModel()
    modifiers = [_modifier(["nonexistent_a"]), _modifier(["layer1", "nonexistent_b"])]

    assert get_unmatched_ignore_entries(model, modifiers) == [
        ("QuantizationModifier", "nonexistent_a"),
        ("QuantizationModifier", "nonexistent_b"),
    ]


@pytest.mark.unit
def test_an_empty_recipe_reports_nothing():
    model = DummyModel()

    assert get_unmatched_ignore_entries(model, []) == []
    assert get_unmatched_ignore_entries(model, [_modifier([])]) == []


@pytest.mark.unit
def test_warning_names_the_entry_and_the_modifier(warning_logs):
    model = DummyModel()
    modifier = _modifier(["layer1", "nonexistent_module"])

    warn_on_unmatched_ignore_entries(model, [modifier])

    assert len(warning_logs) == 1
    assert "nonexistent_module" in warning_logs[0]
    assert "QuantizationModifier" in warning_logs[0]
    # the entry which did match is not mentioned
    assert "layer1" not in warning_logs[0]


@pytest.mark.unit
def test_a_fully_matching_recipe_warns_not_at_all(warning_logs):
    model = DummyModel()
    modifier = _modifier(["layer1", "layer2"])

    warn_on_unmatched_ignore_entries(model, [modifier])

    assert warning_logs == []


@pytest.fixture
def session_with_recipe():
    """Install a recipe on the active session, and restore it afterwards."""

    def install(modifiers):
        active_session().lifecycle.recipe = Recipe(modifiers=modifiers)

    previous = active_session().lifecycle.recipe
    yield install
    active_session().lifecycle.recipe = previous


def _save_and_reload(model, save_path):
    modify_save_pretrained(model)
    model.save_pretrained(save_path, safe_serialization=True)
    return AutoModelForCausalLM.from_pretrained(save_path)


@pytest.mark.integration
def test_save_warns_on_unmatched_ignore_entry(
    tmp_path, warning_logs, session_with_recipe
):
    """The entry is reported at save, and the checkpoint is otherwise unchanged."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
    session_with_recipe([_modifier(["re:.*nonexistent_layer.*"])])

    reloaded = _save_and_reload(model, tmp_path / "save_path")

    assert sum("nonexistent_layer" in log for log in warning_logs) == 1

    # the checkpoint is unchanged: warning only, nothing about the save differs
    model_dict = get_state_dict_offloaded_model(model)
    reloaded_dict = get_state_dict_offloaded_model(reloaded)
    assert model_dict.keys() == reloaded_dict.keys()
    for key in model_dict:
        assert torch.equal(model_dict[key].cpu(), reloaded_dict[key].cpu())


@pytest.mark.integration
def test_save_does_not_warn_when_ignore_entries_match(
    tmp_path, warning_logs, session_with_recipe
):
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
    session_with_recipe([_modifier(["re:.*lm_head.*", "LlamaRMSNorm"])])

    _save_and_reload(model, tmp_path / "save_path")

    assert [log for log in warning_logs if "matched no module" in log] == []
