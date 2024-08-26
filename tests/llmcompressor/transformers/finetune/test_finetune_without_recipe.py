import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_generic"


@pytest.mark.integration
@requires_torch
@requires_gpu
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestFinetuneWithoutRecipe(unittest.TestCase):
    model = None
    dataset = None

    def setUp(self):
        self.output = "./finetune_output"

    def test_finetune_without_recipe(self):
        from llmcompressor.transformers import train

        recipe_str = None
        device = "cuda:0"

        concatenate_data = False
        max_steps = 50
        splits = "train"

        train(
            model=self.model,
            dataset=self.dataset,
            output_dir=self.output,
            recipe=recipe_str,
            max_steps=max_steps,
            concatenate_data=concatenate_data,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
