import os
import shutil
from pathlib import Path

import numpy
import pytest
import yaml
from loguru import logger

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.examples.utils import requires_gpu_count

try:
    import lm_eval

    lm_eval_installed = True
except ImportError:
    lm_eval_installed = False
    logger.warning("lm_eval is not installed. This test will be skipped")

TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", None)


# Will run each test case in its own process through run_tests.sh
# emulating vLLM CI testing
@requires_gpu_count(1)
@pytest.mark.skipif(
    not lm_eval_installed, reason="lm eval is not installed, skipping test"
)
class TestLMEval:
    """
    The following test quantizes a model using a preset scheme or recipe,
    and then evaluates the model using LM Eval. Each test case is focused on a
    specific quantization type (e.g W4A16 with grouped quantization,
    W4N16 with channel quantization). To add a new test case, a new config has to be
    added to the lm_eval_configs folder. The tests run on a cadence defined by the
    `cadence` field. Each config defines the model to quantize. Optionally, a dataset
    id and split can be provided for calibration. Finally, all config files must list
    a scheme. The scheme can be a preset scheme from
    https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py
    or another identifier which can be used for the particular test case. If a recipe
    is not provided, it is assumed that the scheme provided is a preset scheme and will
    be used for quantization. Otherwise, the recipe will always be used if given.
    """  # noqa: E501

    def set_up(self):
        eval_config = yaml.safe_load(Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

        if os.environ.get("CADENCE", "commit") != eval_config.get("cadence"):
            pytest.skip("Skipping test; cadence mismatch")

        self.model = eval_config["model"]
        self.scheme = eval_config.get("scheme")
        self.dataset_id = eval_config.get("dataset_id")
        self.dataset_config = eval_config.get("dataset_config")
        self.dataset_split = eval_config.get("dataset_split")
        self.recipe = eval_config.get("recipe")
        self.quant_type = eval_config.get("quant_type")
        self.save_dir = eval_config.get("save_dir")
        self.task = eval_config.get("task")
        self.num_fewshot = eval_config.get("num_fewshot")
        self.limit = eval_config.get("limit")
        self.exact_flex = eval_config.get("exact_match,flexible-extract")
        self.exact_strict = eval_config.get("exact_match,strict-match")

        logger.info("========== RUNNING ==============")
        logger.info(self.scheme)

        self.device = "cuda:0"
        self.num_calibration_samples = 512
        self.max_seq_length = 2048

    def test_lm_eval(self):
        # Run vLLM with saved model
        self.set_up()
        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
        oneshot_model, tokenizer = run_oneshot_for_e2e_testing(
            model=self.model,
            device=self.device,
            num_calibration_samples=self.num_calibration_samples,
            max_seq_length=self.max_seq_length,
            scheme=self.scheme,
            dataset_id=self.dataset_id,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            recipe=self.recipe,
            quant_type=self.quant_type,
        )

        logger.info("================= SAVING TO DISK ======================")
        oneshot_model.save_pretrained(self.save_dir)
        tokenizer.save_pretrained(self.save_dir)
        recipe_path = os.path.join(self.save_dir, "recipe.yaml")

        # Use the session to fetch the recipe;
        # Reset session for next test case
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)
        session.reset()

        logger.info("================= Running LM Eval ======================")

        model_args = f"pretrained={self.save_dir},add_bos_token=True"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[self.task],
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            device="cuda:0",
            batch_size=100,
        )

        metrics = results["results"][self.task]
        exact_match_strict = metrics.get("exact_match,strict-match")
        exact_match_flex = metrics.get("exact_match,flexible-extract")
        logger.info("Exact Match, Strict")
        logger.info(exact_match_strict)
        logger.info("Exact Match, Flex")
        logger.info(exact_match_flex)
        assert numpy.isclose(exact_match_strict, self.exact_strict, rtol=0.05)
        assert numpy.isclose(exact_match_flex, self.exact_flex, rtol=0.05)

    def tear_down(self):
        if self.save_dir is not None:
            shutil.rmtree(self.save_dir)
