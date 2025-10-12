import os
import random
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy
import pandas as pd
import pytest
import torch
import yaml
from loguru import logger
from pydantic import BaseModel

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.test_timer.timer_utils import get_singleton_manager, log_time
from tests.testing_utils import requires_gpu


class LmEvalConfig(BaseModel):
    model: str = "hf"
    model_args: dict = {"add_bos_token": True, "dtype": "bfloat16"}
    task: str = "gsm8k"
    num_fewshot: int = 5
    limit: int = 1000
    batch_size: int = 100
    # Recovery testing (default): compare against base model performance
    # Default threshold is 0.95 (retain ≥95% of base), can be overridden
    recovery_threshold: Union[float, dict] = 0.95
    # Optional absolute metrics for warnings (not failures)
    metrics: Optional[dict] = None


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
@requires_gpu(1)
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

    Recovery Testing (DEFAULT):
    Tests now use recovery-based validation by default, comparing compressed model
    performance against the base model. Default threshold is 0.95 (≥95% recovery).

    Config options:
    - recovery_threshold: 0.95 (default if not specified)
    - recovery_threshold: 0.93 (override default globally)
    - recovery_threshold: {"metric1": 0.95, "metric2": 0.90} (per-metric)
    - metrics: {...} (optional - used for warnings only, not failures)
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
        logger.info(
            f"Recovery threshold: {self.lmeval.recovery_threshold} (default: 0.95)"
        )
        if self.lmeval.metrics:
            logger.info("Absolute metrics provided - will show warnings if outside ±5%")

        self.num_calibration_samples = eval_config.get("num_calibration_samples", 512)
        self.max_seq_length = 2048

    def test_lm_eval(self, test_data_file: str):
        # Run vLLM with saved model
        self.set_up(test_data_file)

        # Always evaluate base model for recovery testing
        logger.info("================= Evaluating BASE model ======================")
        self.base_results = self._eval_base_model()

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

        logger.info("================= Running LM Eval on COMPRESSED model ==========")
        self._run_lm_eval()

        self.tear_down()

    @log_time
    def _eval_base_model(self):
        """Evaluate the base (uncompressed) model."""
        model_args = {**self.lmeval.model_args, "pretrained": self.model}

        results = lm_eval.simple_evaluate(
            model=self.lmeval.model,
            model_args=model_args,
            tasks=[self.lmeval.task],
            num_fewshot=self.lmeval.num_fewshot,
            limit=self.lmeval.limit,
            device="cuda:0",
            batch_size=self.lmeval.batch_size,
        )

        return results

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

        # Always use recovery testing
        self._validate_recovery(results)

        # If absolute metrics provided, show warnings (not failures)
        if self.lmeval.metrics:
            self._check_absolute_warnings(results)

    def _validate_recovery(self, compressed_results):
        """Validate using recovery testing - compare against base model."""
        base_metrics = self.base_results["results"][self.lmeval.task]
        compressed_metrics = compressed_results["results"][self.lmeval.task]
        higher_is_better_map = compressed_results.get("higher_is_better", {}).get(
            self.lmeval.task, {}
        )

        logger.info("=" * 80)
        logger.info("RECOVERY TESTING COMPARISON")
        logger.info("=" * 80)

        # Get default threshold from config schema
        default_threshold = self.lmeval.model_fields["recovery_threshold"].default

        failures = []
        # Iterate over compressed metrics (what we actually got)
        for metric_key, compressed_val in compressed_metrics.items():
            # Skip stderr and other metadata
            if "stderr" in metric_key or metric_key.startswith("alias"):
                continue

            base_val = base_metrics.get(metric_key)
            if base_val is None:
                logger.warning(
                    f"Metric {metric_key} in compressed results "
                    f"not found in base results, skipping"
                )
                continue

            # Get threshold for this metric
            if isinstance(self.lmeval.recovery_threshold, dict):
                threshold = self.lmeval.recovery_threshold.get(
                    metric_key, default_threshold
                )
            else:
                threshold = self.lmeval.recovery_threshold

            # Get direction
            base_metric_name = metric_key.split(",")[0]
            higher_is_better = higher_is_better_map.get(base_metric_name, True)

            # Compute recovery
            if base_val == 0:
                recovery = 1.0 if compressed_val == 0 else 0.0
            elif higher_is_better:
                recovery = compressed_val / base_val
            else:
                # For "lower is better", invert ratio
                recovery = base_val / compressed_val

            # Check threshold
            passed = recovery >= threshold
            direction = "↑" if higher_is_better else "↓"

            msg = (
                f"{metric_key:40} | Base: {base_val:.4f} | "
                f"Compressed: {compressed_val:.4f} | "
                f"Recovery: {recovery:6.2%} {direction} | Threshold: ≥{threshold:.2%}"
            )

            if passed:
                logger.info(f"✓ {msg}")
            else:
                logger.error(f"✗ {msg}")
                failures.append(
                    f"{metric_key}: {recovery:.2%} < {threshold:.2%} "
                    f"(base={base_val:.4f}, compressed={compressed_val:.4f})"
                )

        # Validate that config thresholds match actual results
        if isinstance(self.lmeval.recovery_threshold, dict):
            for config_metric_key in self.lmeval.recovery_threshold.keys():
                if config_metric_key not in compressed_metrics:
                    logger.warning(
                        f"Metric {config_metric_key} in recovery_threshold config "
                        f"not found in results"
                    )

        logger.info("=" * 80)

        if failures:
            failure_msg = "\n".join(failures)
            raise AssertionError(f"Recovery testing failed:\n{failure_msg}")

        logger.info("✓ ALL METRICS PASSED RECOVERY THRESHOLDS")
        logger.info("=" * 80)

    def _check_absolute_warnings(self, results):
        """Check absolute metrics and warn if outside ±5% tolerance (not a failure)."""
        logger.info("=" * 80)
        logger.info("ABSOLUTE METRICS CHECK (warnings only, not failures)")
        logger.info("=" * 80)

        metrics: dict = results["results"][self.lmeval.task]
        for metric_key, expected_val in self.lmeval.metrics.items():
            # Skip stderr metrics
            if "stderr" in metric_key:
                continue

            actual_val = metrics.get(metric_key)
            if actual_val is None:
                logger.warning(
                    f"Metric {metric_key} in config not found in results, "
                    f"skipping warning check"
                )
                continue

            higher_is_better = (
                results.get("higher_is_better", {})
                .get(self.lmeval.task, {})
                .get(metric_key.split(",")[0], True)
            )

            # Check if within ±5% relative tolerance
            lower_bound = expected_val * 0.95
            upper_bound = expected_val * 1.05

            if higher_is_better:
                # For higher is better, we care about lower bound
                if actual_val < lower_bound:
                    logger.warning(
                        f"⚠ {metric_key:40} | Expected: {expected_val:.4f} (±5%) | "
                        f"Got: {actual_val:.4f} | Below expected range"
                    )
            else:
                # For lower is better, we care about upper bound
                if actual_val > upper_bound:
                    logger.warning(
                        f"⚠ {metric_key:40} | Expected: {expected_val:.4f} (±5%) | "
                        f"Got: {actual_val:.4f} | Above expected range"
                    )

        logger.info("=" * 80)

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
