import os
import shutil
import unittest
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot, train
from llmcompressor.core import create_session
from llmcompressor.modifiers.quantization import QuantizationModifier


@pytest.mark.unit
class TestOneshotThenFinetune(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")
        self.quantization_config = CompressedTensorsConfig(run_compressed=False)

    def test_oneshot_sparsification_then_finetune(self):
        recipe_str = "tests/llmcompressor/transformers/obcq/recipes/test_tiny2.yaml"
        model = AutoModelForCausalLM.from_pretrained(
            "nm-testing/llama2.c-stories15M", torch_dtype="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:5%]"}

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

        # Explictly decompress the model for training using quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            self.output / "oneshot_out",
            torch_dtype="auto",
            quantization_config=self.quantization_config,
        )
        distill_teacher = AutoModelForCausalLM.from_pretrained(
            "nm-testing/llama2.c-stories15M", torch_dtype="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        output_dir = self.output / "finetune_out"
        splits = "train[5%:7%]"

        recipe = """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    targets: ['re:.*q_proj.weight', 're:.*k_proj.weight',
                    're:.*v_proj.weight', 're:.*o_proj.weight',
                    're:.*gate_proj.weight', 're:.*up_proj.weight',
                    're:.*down_proj.weight']
                    start: 0
        """

        with create_session():
            train(
                model=model,
                distill_teacher=distill_teacher,
                dataset=dataset,
                output_dir=output_dir,
                num_train_epochs=0.05,
                concatenate_data=concatenate_data,
                splits=splits,
                recipe=recipe,
            )

        # test reloading checkpoint and final model
        # verify checkpoint reloading and can carry out finetune
        # with the saved model
        # Explictly decompress the model for training using quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            quantization_config=self.quantization_config,
        )

        with create_session():
            train(
                model=model,
                distill_teacher=distill_teacher,
                dataset=dataset,
                output_dir=output_dir,
                num_train_epochs=0.05,
                concatenate_data=concatenate_data,
                splits=splits,
                recipe=recipe,
                resume_from_checkpoint=True,  # use last checkpoint
            )

    def test_oneshot_quantization_then_finetune(self):
        recipe = QuantizationModifier(
            targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
        )

        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:5%]"}

        with create_session():
            oneshot(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe,
                concatenate_data=concatenate_data,
                splits=splits,
            )

        from transformers.utils.quantization_config import CompressedTensorsConfig

        quantization_config = CompressedTensorsConfig(run_compressed=False)
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )

        dataset = "open_platypus"
        concatenate_data = False
        output_dir = self.output / "finetune_out"
        splits = {"calibration": "train[:5%]", "train": "train[5%:7%]"}

        with create_session():
            train(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                concatenate_data=concatenate_data,
                splits=splits,
                num_train_epochs=0.05,
            )

        # test reloading checkpoint and final model
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )

        with create_session():
            train(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                concatenate_data=concatenate_data,
                splits=splits,
                num_train_epochs=0.05,
                resume_from_checkpoint=True,  # use last checkpoint
            )

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
