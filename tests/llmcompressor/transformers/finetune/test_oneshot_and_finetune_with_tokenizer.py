import shutil
import unittest

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.skip(reason="slow")
class TestOneshotAndFinetuneWithTokenizer(unittest.TestCase):
    def setUp(self):
        self.output = "./finetune_output"

    def test_oneshot_and_finetune_with_tokenizer(self):
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from llmcompressor.transformers import SparseAutoModelForCausalLM, compress

        recipe_str = (
            "tests/llmcompressor/transformers/finetune/test_alternate_recipe.yaml"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Xenova/llama2.c-stories15M",
        )
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        model = SparseAutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map=device
        )

        dataset_config_name = "wikitext-2-raw-v1"
        dataset = load_dataset("wikitext", dataset_config_name, split="train[:50%]")

        concatenate_data = True
        run_stages = True
        max_steps = 50
        splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

        compress(
            model=model,
            dataset=dataset,
            dataset_config_name=dataset_config_name,
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
