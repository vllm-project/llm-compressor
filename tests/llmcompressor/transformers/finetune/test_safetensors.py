import os
import shutil
import unittest
from pathlib import Path

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestSafetensors(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_safetensors(self):
        import torch

        from llmcompressor.transformers import train

        model = "Xenova/llama2.c-stories15M"
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"

        dataset = "open_platypus"
        output_dir = self.output / "output1"
        max_steps = 10
        splits = {"train": "train[:10%]"}

        train(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            splits=splits,
            oneshot_device=device,
        )

        assert os.path.exists(output_dir / "model.safetensors")
        assert not os.path.exists(output_dir / "pytorch_model.bin")

        # test we can also load
        new_output_dir = self.output / "output2"
        train(
            model=output_dir,
            dataset=dataset,
            output_dir=new_output_dir,
            max_steps=max_steps,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
