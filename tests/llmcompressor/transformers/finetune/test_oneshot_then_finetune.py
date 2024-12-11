import os
import shutil
import unittest
from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.skipif(
    "CADENCE" in os.environ
    and (os.environ["CADENCE"] == "weekly" or os.environ["CADENCE"] == "nightly"),
    reason="Don't run for weekly and nightly tests as those use multi gpu "
    "runners and this test fails when ngpu>1",
)
class TestOneshotThenFinetune(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_oneshot_then_finetune(self):
        from transformers import AutoModelForCausalLM

        from llmcompressor.core import create_session
        from llmcompressor.transformers import oneshot, train

        raise Exception(
            "This test is not supported until we can decompress sparse"
            "models/turn off run_compressed through transformers"
        )

        recipe_str = "tests/llmcompressor/transformers/obcq/recipes/test_tiny2.yaml"
        model = AutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:10%]"}

        with create_session():
            oneshot(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe_str,
                concatenate_data=concatenate_data,
                splits=splits,
            )

        recipe_str = (
            "tests/llmcompressor/transformers/finetune/test_finetune_recipe.yaml"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.output / "oneshot_out", device_map="auto"
        )
        distill_teacher = AutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map="auto"
        )

        dataset = "open_platypus"
        concatenate_data = False
        output_dir = self.output / "finetune_out"
        splits = "train[:50%]"
        max_steps = 25

        with create_session():
            train(
                model=model,
                distill_teacher=distill_teacher,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe_str,
                concatenate_data=concatenate_data,
                splits=splits,
                max_steps=max_steps,
            )

        # test reloading checkpoint and final model
        model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")

        with create_session():
            train(
                model=model,
                distill_teacher=distill_teacher,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe_str,
                concatenate_data=concatenate_data,
                splits=splits,
                max_steps=max_steps,
                resume_from_checkpoint=True,  # use last checkpoint
            )

    def tearDown(self):
        shutil.rmtree(self.output)
