from pathlib import Path

import pytest

from llmcompressor.core.session_functions import reset_session
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.smoothquant.base import DEFAULT_SMOOTHQUANT_MAPPINGS
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from tests.testing_utils import requires_gpu


@pytest.fixture
def common_setup():
    model_stub = "Xenova/llama2.c-stories15M"
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, device_map="auto", torch_dtype="auto"
    )

    dataset = "ultrachat-200k"
    output_dir = "./test_output"
    splits = {"calibration": "train_gen[:5%]"}
    max_seq_length = 2048
    pad_to_max_length = False
    num_calibration_samples = 8

    return (
        model,
        dataset,
        output_dir,
        splits,
        max_seq_length,
        pad_to_max_length,
        num_calibration_samples,
    )


def recipes():
    modifier_objects = [
        SmoothQuantModifier(
            smoothing_strength=0.8, mappings=DEFAULT_SMOOTHQUANT_MAPPINGS
        ),
        GPTQModifier(
            targets="Linear", scheme="W8A8", ignore=["lm_head"], sequential_update=False
        ),
    ]

    recipe_str = """
DEFAULT_stage:
  DEFAULT_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.8
      mappings:
      - - ['re:.*q_proj', 're:.*k_proj', 're:.*v_proj']
        - re:.*input_layernorm
      - - ['re:.*gate_proj', 're:.*up_proj']
        - re:.*post_attention_layernorm
    GPTQModifier:
      sequential_update: false
      targets: Linear
      scheme: W8A8
"""

    recipe_file = str(Path(__file__).parent / "recipe.yaml")

    return [modifier_objects, recipe_str, recipe_file]


@requires_gpu
@pytest.mark.parametrize("recipe", recipes())
def test_oneshot(common_setup, recipe):
    (
        model,
        dataset,
        output_dir,
        splits,
        max_seq_length,
        pad_to_max_length,
        num_calibration_samples,
    ) = common_setup

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        splits=splits,
        max_seq_length=max_seq_length,
        pad_to_max_length=pad_to_max_length,
        num_calibration_samples=num_calibration_samples,
        save_compressed=True,
    )

    reset_session()
