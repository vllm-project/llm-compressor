from types import SimpleNamespace

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

import llmcompressor.pipelines.independent.pipeline as independent_module
from llmcompressor.core import State
from llmcompressor.modifiers.transform import SpinQuantModifier


@pytest.mark.unit
def test_rotation_restructures_are_scoped_to_spinquant_pass(monkeypatch):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    state = State(model=model)
    spinquant = SpinQuantModifier(rotations=["R1"], transform_type="hadamard")
    spinquant.on_initialize(state)
    modifiers = [object(), spinquant, object()]
    session = SimpleNamespace(
        state=state,
        lifecycle=SimpleNamespace(recipe=SimpleNamespace(modifiers=modifiers)),
    )
    observed = []

    class RecordingPipeline:
        def __call__(self, model, dataloader, dataset_args):
            observed.append(state.rotation_restructures)

    monkeypatch.setattr(independent_module, "active_session", lambda: session)
    monkeypatch.setattr(
        independent_module.CalibrationPipeline,
        "from_modifiers",
        lambda modifiers: RecordingPipeline(),
    )

    independent_module.IndependentPipeline()(model, None, None)

    assert observed == [False, True, False]
    assert state.rotation_restructures is True
    assert session.lifecycle.recipe.modifiers is modifiers
