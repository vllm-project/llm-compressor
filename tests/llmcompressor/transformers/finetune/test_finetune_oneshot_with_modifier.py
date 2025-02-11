import shutil
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_generic"


@pytest.mark.integration
@requires_gpu
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotWithModifierObject(unittest.TestCase):
    model = None
    dataset = None

    def setUp(self):
        self.output = Path("./finetune_output")

    def test_post_train_with_modifier_object(self):
        from llmcompressor import post_train
        from llmcompressor.modifiers.obcq.base import SparseGPTModifier

        recipe_str = [
            SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])
        ]

        device = "cuda:0"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "post_train_out"
        splits = {"calibration": "train[:10%]"}

        post_train(
            model=self.model,
            dataset=self.dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
            post_train_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
