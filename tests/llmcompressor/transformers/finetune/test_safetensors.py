import os
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
class TestSafetensors(unittest.TestCase):
    model = None
    dataset = None

    def setUp(self):
        self.output = Path("./finetune_output")

    def test_safetensors(self):
        from llmcompressor import train

        output_dir = self.output / "output1"
        max_steps = 10
        splits = {"train": "train[:10%]"}

        train(
            model=self.model,
            dataset=self.dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            splits=splits,
        )

        assert os.path.exists(output_dir / "model.safetensors")
        assert not os.path.exists(output_dir / "pytorch_model.bin")

        # test we can also load
        new_output_dir = self.output / "output2"
        train(
            model=output_dir,
            dataset=self.dataset,
            output_dir=new_output_dir,
            max_steps=max_steps,
            splits=splits,
        )

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
