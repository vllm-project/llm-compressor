from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.core.session_functions import reset_session
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.smoothquant.utils import DEFAULT_SMOOTHQUANT_MAPPINGS
from tests.testing_utils import requires_gpu


@pytest.fixture
def setup_model_and_config(tmp_path):
    """
    Loads a test model and returns common arguments used in oneshot runs.
    """
    model = AutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories110M",
        torch_dtype="auto",
    )

    return {
        "model": model,
        "dataset": "ultrachat-200k",
        "output_dir": str(tmp_path / "compressed_output"),
        "splits": {"calibration": "train_gen[:10]"},
        "max_seq_length": 2048,
        "pad_to_max_length": False,
        "num_calibration_samples": 8,
    }


def recipe_variants():
    """
    Returns three ways of defining the compression recipe:
    - Python modifier objects
    - YAML string
    - Path to a YAML file
    """
    python_modifiers = [
        SmoothQuantModifier(
            smoothing_strength=0.8,
            mappings=DEFAULT_SMOOTHQUANT_MAPPINGS,
        ),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head"],
        ),
    ]

    yaml_string = """
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
      targets: Linear
      ignore: [lm_head]
      scheme: W8A8
"""

    yaml_path = str(Path(__file__).parent / "recipe.yaml")

    return [python_modifiers, yaml_string, yaml_path]


@requires_gpu
@pytest.mark.regression
@pytest.mark.parametrize(
    "recipe",
    recipe_variants(),
    ids=["modifier-objects", "yaml-string", "yaml-file"],
)
def test_oneshot_accepts_multiple_recipe_formats(setup_model_and_config, recipe):
    """
    Integration test: verifies oneshot runs successfully with different recipe formats.
    """
    oneshot(
        recipe=recipe,
        **setup_model_and_config,
        save_compressed=True,
    )

    output_path = Path(setup_model_and_config["output_dir"])
    assert output_path.exists() and any(
        output_path.iterdir()
    ), f"No output artifacts found in: {output_path}"

    reset_session()
