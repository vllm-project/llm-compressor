import os

import pytest

from llmcompressor import train
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_generic"


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_safetensors(config, tmp_path):
    model = config["model"]
    dataset = config["dataset"]
    output = tmp_path / "finetune_output"

    output_dir = output / "output1"
    max_steps = 10
    splits = {"train": "train[:10%]"}

    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        max_steps=max_steps,
        splits=splits,
    )

    assert os.path.exists(output_dir / "model.safetensors")
    assert not os.path.exists(output_dir / "pytorch_model.bin")

    # test we can also load
    new_output_dir = output / "output2"
    train(
        model=output_dir,
        dataset=dataset,
        output_dir=new_output_dir,
        max_steps=max_steps,
        splits=splits,
    )
