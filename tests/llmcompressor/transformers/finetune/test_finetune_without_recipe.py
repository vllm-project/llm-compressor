import shutil
import unittest

import pytest

from tests.testing_utils import requires_torch, @requires_gpu

@pytest.mark.skip(reason="slow")
class TestFinetuneWithoutRecipe(unittest.TestCase):
    def setUp(self):
        self.output = "./finetune_output"

    def test_finetune_without_recipe(self):
        import torch

        from llmcompressor.transformers import train

        recipe_str = None
        model = "Xenova/llama2.c-stories15M"
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        dataset = "open_platypus"
        concatenate_data = False
        max_steps = 50
        splits = "train"

        train(
            model=model,
            dataset=dataset,
            output_dir=self.output,
            recipe=recipe_str,
            max_steps=max_steps,
            concatenate_data=concatenate_data,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
