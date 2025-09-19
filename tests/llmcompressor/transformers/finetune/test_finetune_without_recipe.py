import pytest

from llmcompressor import train
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_generic"


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_finetune_without_recipe(config, tmp_path):
    model = config["model"]
    dataset = config["dataset"]
    output = tmp_path / "finetune_output"

    recipe_str = None

    concatenate_data = False
    max_steps = 50
    splits = "train"

    train(
        model=model,
        dataset=dataset,
        output_dir=output,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
    )
