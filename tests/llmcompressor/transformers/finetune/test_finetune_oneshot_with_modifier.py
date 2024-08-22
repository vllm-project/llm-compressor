import shutil
import unittest
from pathlib import Path

import pytest

from tests.testing_utils import requires_torch, requires_gpu

@pytest.mark.integration
@requires_gpu
@requires_torch
class TestOneshotWithModifierObject(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_oneshot_with_modifier_object(self):
        import torch

        from llmcompressor.modifiers.obcq.base import SparseGPTModifier
        from llmcompressor.transformers import oneshot

        recipe_str = [
            SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])
        ]
        model = "Xenova/llama2.c-stories15M"
        device = "cuda:0"
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:10%]"}

        oneshot(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
