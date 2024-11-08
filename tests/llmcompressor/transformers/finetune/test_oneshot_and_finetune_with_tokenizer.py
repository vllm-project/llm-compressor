import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_tokenizer"


@pytest.mark.integration
@requires_torch
@requires_gpu
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotAndFinetuneWithTokenizer(unittest.TestCase):
    model = None
    dataset = None
    dataset_config_name = None

    def setUp(self):
        self.output = "./finetune_output"

    def test_oneshot_and_finetune_with_tokenizer(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from llmcompressor.transformers import SparseAutoModelForCausalLM, compress

        recipe_str = (
            "tests/llmcompressor/transformers/finetune/test_alternate_recipe.yaml"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
        )

        model_loaded = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto"
        )

        dataset_loaded = load_dataset(
            self.dataset, self.dataset_config_name, split="train[:50%]"
        )

        concatenate_data = True
        run_stages = True
        max_steps = 50
        splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

        compress(
            model=model_loaded,
            dataset=dataset_loaded,
            dataset_config_name=self.dataset_config_name,
            run_stages=run_stages,
            output_dir=self.output,
            recipe=recipe_str,
            max_steps=max_steps,
            concatenate_data=concatenate_data,
            splits=splits,
            tokenizer=tokenizer,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
