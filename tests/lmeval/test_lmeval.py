import os
import random
import shutil
from pathlib import Path

import numpy
import pandas as pd
import pytest
import torch
import yaml
from loguru import logger
from pydantic import BaseModel

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.examples.utils import requires_gpu_count
from tests.test_timer.timer_utils import get_singleton_manager, log_time


class LmEvalConfig(BaseModel):
    model: str = "hf"
    model_args: dict = {"add_bos_token": True, "dtype": "bfloat16"}
    task: str = "gsm8k"
    num_fewshot: int = 5
    limit: int = 1000
    metrics: dict
    batch_size: int = 100


try:
    import lm_eval

    lm_eval_installed = True
except ImportError:
    lm_eval_installed = False
    logger.warning("lm_eval is not installed. This test will be skipped")

TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", None)
TIMINGS_DIR = os.environ.get("TIMINGS_DIR", "timings/lm-eval")


# Will run each test case in its own process through run_tests.sh
# emulating vLLM CI testing
@requires_gpu_count(1)
@pytest.mark.parametrize(
    "test_data_file", [pytest.param(TEST_DATA_FILE, id=TEST_DATA_FILE)]
)
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

    def set_up(self, test_data_file: str):
        eval_config = yaml.safe_load(Path(test_data_file).read_text(encoding="utf-8"))

        if os.environ.get("CADENCE", "commit") != eval_config.get("cadence"):
            pytest.skip("Skipping test; cadence mismatch")

        self.model = eval_config["model"]
        self.model_class = eval_config.get("model_class", "AutoModelForCausalLM")
        self.lmeval = LmEvalConfig(**eval_config.get("lmeval", {}))
        self.scheme = eval_config.get("scheme")
        self.dataset_id = eval_config.get("dataset_id")
        self.dataset_config = eval_config.get("dataset_config")
        self.dataset_split = eval_config.get("dataset_split")
        self.recipe = eval_config.get("recipe")
        self.quant_type = eval_config.get("quant_type")
        self.save_dir = eval_config.get("save_dir")

        seed = eval_config.get("seed", None)
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

        logger.info("========== RUNNING ==============")
        logger.info(self.scheme)

        self.num_calibration_samples = 512
        self.max_seq_length = 2048

    def test_lm_eval(self, test_data_file: str):
        # Run vLLM with saved model
        self.set_up(test_data_file)

        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
        oneshot_model, processor = run_oneshot_for_e2e_testing(
            model=self.model,
            model_class=self.model_class,
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
        self._save_compressed_model(oneshot_model, processor)

        # Use the session to fetch the recipe;
        # Reset session for next test case
        self._handle_recipe()

        logger.info("================= Running LM Eval ======================")
        self._run_lm_eval()

        self.tear_down()

    @log_time
    def _save_compressed_model(self, oneshot_model, processor):
        oneshot_model.save_pretrained(self.save_dir)
        processor.save_pretrained(self.save_dir)

    @log_time
    def _handle_recipe(self):
        recipe_path = os.path.join(self.save_dir, "recipe.yaml")
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)
        session.reset()

    @log_time
    def _run_lm_eval(self):
        model_args = {"pretrained": self.save_dir}
        model_args.update(self.lmeval.model_args)
        results = lm_eval.simple_evaluate(
            model=self.lmeval.model,
            model_args=model_args,
            tasks=[self.lmeval.task],
            num_fewshot=self.lmeval.num_fewshot,
            limit=self.lmeval.limit,
            device="cuda:0",
            batch_size=self.lmeval.batch_size,
        )

        metrics: dict = results["results"][self.lmeval.task]
        for metric_key, expected_val in self.lmeval.metrics.items():
            # stderr metrics are only used as absolute tolerance
            # checks for actual values
            if "stderr" in metric_key:
                continue
            actual_val = metrics.get(metric_key)
            higher_is_better = results["higher_is_better"][self.lmeval.task].get(
                metric_key.split(",")[0], True
            )
            stderr_key = metric_key.replace(",", "_stderr,")
            std_err = self.lmeval.metrics.get(stderr_key)

            # If stderr is provided, use it as absolute tolerance
            # Otherwise, default to a 5% relative tolerance
            if std_err is None:
                logger.info(
                    f"Comparing {metric_key}: Expecting {expected_val} "
                    f"relative tolerance ±5%, Got {actual_val}. "
                    f"Higher is better: {higher_is_better}"
                )
                # If higher is better, assert actual val >= expected val * (1 - stderr)
                if higher_is_better:
                    assert actual_val >= expected_val * (0.95)
                # If higher is worse, assert actual val <= expected val * (1 + stderr)
                else:
                    assert actual_val <= expected_val * (1.05)

            else:
                logger.info(
                    f"Comparing {metric_key}: Expecting {expected_val} "
                    f"absolute tolerance ±{std_err*100}%, Got {actual_val}. "
                    f"Higher is better: {higher_is_better}"
                )
                # If higher is better, assert actual val >= expected val - stderr
                if higher_is_better:
                    assert actual_val >= expected_val - std_err
                # If higher is worse, assert actual val <= expected val + stderr
                else:
                    assert actual_val <= expected_val + std_err

    def tear_down(self):
        timer = get_singleton_manager()
        # fetch dictionary of measurements, where keys are func names
        # and values are the time it took to run the method, each
        # time it was called
        measurements = timer.measurements
        if measurements:
            p = Path(TIMINGS_DIR)
            p.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(measurements)
            df.to_csv(p / f"{self.save_dir}.csv", index=False)

        if self.save_dir is not None and os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
