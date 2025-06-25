import os
import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_generic"


@pytest.mark.integration
@requires_gpu
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestFinetuneWithoutRecipe(unittest.TestCase):
    model = None
    dataset = None

    def setUp(self):
        self.output = "./finetune_output"

    def test_finetune_without_recipe(self):
        from llmcompressor import train

        recipe_str = None

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
        )

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
