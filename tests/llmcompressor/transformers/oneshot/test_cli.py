import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, run_cli_command

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/oneshot/oneshot_configs"


@pytest.mark.smoke
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneShotCli(unittest.TestCase):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None
    tokenize = None

    def setUp(self):
        if self.tokenize:
            pytest.skip("Tokenized data input not supported for oneshot cli")

        self.output = "./oneshot_output"
        self.additional_args = []
        if self.dataset_config_name:
            self.additional_args.append("--dataset_config_name")
            self.additional_args.append(self.dataset_config_name)

    def test_one_shot_cli(self):
        cmd = [
            "llmcompressor.transformers.text_generation.oneshot",
            "--dataset",
            self.dataset,
            "--model",
            self.model,
            "--output_dir",
            self.output,
            "--recipe",
            self.recipe,
            "--num_calibration_samples",
            "10",
            "--pad_to_max_length",
            "False",
        ]

        if len(self.additional_args) > 0:
            cmd.extend(self.additional_args)

        print(" ".join(cmd))

        res = run_cli_command(cmd)
        self.assertEqual(res.returncode, 0)
        print(res.stdout)

    def tearDown(self):
        shutil.rmtree(self.output)
