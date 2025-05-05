import os
import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.llmcompressor.transformers.oneshot.dataset_processing import get_data_utils
from tests.testing_utils import parse_params

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/oneshot/oneshot_configs"

# TODO: Seems better to mark test type (smoke, sanity, regression) as a marker as
# opposed to using a field in the config file?


@pytest.mark.smoke
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneShotInputs(unittest.TestCase):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None
    tokenize = None

    def setUp(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)
        self.output = "./oneshot_output"
        self.kwargs = {"dataset_config_name": self.dataset_config_name}

        data_utils = get_data_utils(self.dataset)

        def wrapped_preprocess_func(sample):
            preprocess_func = data_utils.get("preprocess")
            return self.tokenizer(
                preprocess_func(sample), padding=False, max_length=512, truncation=True
            )

        # If `tokenize` is set to True, use the appropriate preprocessing function
        # and set self.tokenizer = None. Updates the self.dataset field from the string
        # to the loaded dataset.
        if self.tokenize:
            loaded_dataset = data_utils.get("dataload")()
            self.dataset = loaded_dataset.map(wrapped_preprocess_func)
            self.tokenizer = None

    def test_one_shot_inputs(self):
        from llmcompressor import oneshot

        oneshot(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            recipe=self.recipe,
            output_dir=self.output,
            num_calibration_samples=10,
            pad_to_max_length=False,
            **self.kwargs,
        )

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
