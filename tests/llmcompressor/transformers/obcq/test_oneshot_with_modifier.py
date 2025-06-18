import os
import shutil
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/obcq/obcq_configs/sparsity_generic"
)


@pytest.mark.integration
@requires_gpu
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotWithModifierObject(unittest.TestCase):
    model = None
    dataset = None

    def setUp(self):
        self.output = Path("./finetune_output")

    def test_oneshot_with_modifier_object(self):
        from llmcompressor import oneshot
        from llmcompressor.modifiers.obcq.base import SparseGPTModifier

        recipe_str = [
            SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])
        ]

        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:10%]"}

        oneshot(
            model=self.model,
            dataset=self.dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
        )

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
