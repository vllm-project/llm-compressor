# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import unittest

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestOneshotAndFinetuneWithTokenizer(unittest.TestCase):
    def setUp(self):
        self.output = "./finetune_output"

    def test_oneshot_and_finetune_with_tokenizer(self):
        import torch
        from datasets import load_dataset

        from sparseml.transformers import (
            SparseAutoModelForCausalLM,
            SparseAutoTokenizer,
            compress,
        )

        recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
        model = SparseAutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")
        tokenizer = SparseAutoTokenizer.from_pretrained(
            "Xenova/llama2.c-stories15M",
        )
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"

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
            oneshot_device=device,
            tokenizer=tokenizer,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
