from typing import Any, Dict, Optional, Union

import pytest
from datasets import load_dataset
from torch.nn import Module
from transformers import AutoModelForCausalLM, Trainer

from llmcompressor.core import active_session
from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn


class MixInTest(SessionManagerMixIn, Trainer):
    def __init__(
        self,
        model: Module,
        recipe: Optional[str],
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            recipe=recipe,
            recipe_args=recipe_args,
            teacher=teacher,
            **kwargs,
        )


def test_mixin_init():
    model_state_path = "Xenova/llama2.c-stories15M"
    model = AutoModelForCausalLM.from_pretrained(model_state_path)
    recipe = "tests/llmcompressor/transformers/finetune/test_quantization.yaml"

    session_mixin = MixInTest(model=model, recipe=recipe)
    assert isinstance(session_mixin, SessionManagerMixIn)
    assert isinstance(session_mixin, Trainer)
    assert session_mixin.recipe == recipe
    assert session_mixin.model == model


@pytest.fixture
def mixin_trainer():
    model_state_path = "Xenova/llama2.c-stories15M"
    model = AutoModelForCausalLM.from_pretrained(model_state_path)
    recipe = "tests/llmcompressor/transformers/finetune/test_quantization.yaml"
    train_dataset = load_dataset("garage-bAInd/Open-Platypus", split="train[:5%]")
    eval_dataset = load_dataset("garage-bAInd/Open-Platypus", split="train[5%:6%]")

    return MixInTest(
        model=model,
        recipe=recipe,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def test_mixin_session_init(mixin_trainer):
    mixin_trainer.initialize_session(epoch=0.0, checkpoint=None)
    session = active_session()

    assert not session.lifecycle.initialized_structure
    assert session.lifecycle.initialized_
