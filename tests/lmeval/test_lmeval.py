import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy
import pytest
import torch
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.testing_utils import BaseTestConfig, cached_lm_eval_run, requires_gpu


class LmEvalConfig(BaseModel):
    model: str = "vllm"
    add_bos_token: bool = True
    dtype: str = "bfloat16"
    task: str = "gsm8k"
    num_fewshot: int = 5
    limit: int = 1000
    fewshot_as_multiturn: bool = False
    apply_chat_template: bool = False
    # Recovery testing (default): compare against base model performance
    # Default threshold is 0.95 (retain ≥95% of base), can be overridden
    recovery_threshold: Union[float, dict] = 0.95
    # Optional absolute metrics for warnings (not failures)
    metrics: Optional[dict] = None
    trust_remote_code: bool = False


class TestConfig(BaseTestConfig):
    """
    Configuration for a single lm-eval test case, loaded from a YAML config file.

    Extends BaseTestConfig with lm-evaluation-harness settings.

    LM Eval settings
    ----------------
    lmeval : LmEvalConfig
        Full lm-evaluation-harness configuration (task, shots, limits, thresholds…).
    """

    lmeval: LmEvalConfig = Field(
        default_factory=LmEvalConfig,
        description="LM Eval harness configuration (task, shots, limits, thresholds…)",
    )


TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", None)
test_file_dir = os.path.dirname(os.path.abspath(__file__))


# Will run each test case in its own process through run_tests_in_python.sh
# emulating vLLM CI testing
@requires_gpu(1)
@pytest.mark.parametrize(
    "test_data_file", [pytest.param(TEST_DATA_FILE, id=TEST_DATA_FILE)]
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

        self.config = TestConfig(**eval_config)

        # Derive save_dir from the config filename so there is no dependency on scheme
        if not self.config.save_dir:
            self.config.save_dir = Path(test_data_file).stem

        random.seed(self.config.seed)
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.use_deterministic_algorithms(True)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"Seed set to {self.config.seed} with deterministic mode enabled")

        logger.info("========== RUNNING ==============")
        logger.info(self.config.scheme)
        logger.info(
            f"Recovery threshold: {self.config.lmeval.recovery_threshold} (default: 0.95)"  # noqa: E501
        )
        if self.config.lmeval.metrics:
            logger.info("Absolute metrics provided - will show warnings if outside ±5%")

    def test_lm_eval(self, test_data_file: str):
        # Run vLLM with saved model
        self.set_up(test_data_file)

        # Always evaluate base model for recovery testing
        logger.info("================= Evaluating BASE model ======================")
        base_results = self._eval_base_model()

        oneshot_model, processor = run_oneshot_for_e2e_testing(
            model=self.config.model,
            model_class=self.config.model_class,
            num_calibration_samples=self.config.num_calibration_samples,
            max_seq_length=2048,
            scheme=self.config.scheme,
            dataset_id=self.config.dataset_id,
            dataset_config=self.config.dataset_config,
            dataset_split=self.config.dataset_split,
            recipe=self.config.recipe,
            quant_type=self.config.quant_type,
        )

        logger.info("================= SAVING TO DISK ======================")
        self._save_compressed_model(oneshot_model, processor)

        logger.info("================= Running LM Eval on COMPRESSED model ==========")
        compressed_results = self._eval_compressed_model()

        # Always use recovery testing
        self._validate_recovery(base_results, compressed_results)

        # If absolute metrics provided, show warnings (not failures)
        if self.config.lmeval.metrics:
            self._check_absolute_warnings(compressed_results)

        self.tear_down()

    @cached_lm_eval_run
    def _eval_base_model(self) -> dict:
        """Evaluate the base (uncompressed) model with caching."""
        return self._eval_model_with_vllm(self.config.model)

    def _eval_compressed_model(self) -> dict:
        """Evaluate the compressed model."""
        return self._eval_model_with_vllm(self.config.save_dir)

    def _eval_model_with_vllm(self, model: str) -> dict:
        run_file_path = os.path.join(test_file_dir, "run_lmeval.py")
        logger.info("Run vllm in subprocess.Popen using python env:")
        logger.info(self.vllm_env)
        # Ensure the venv's bin dir is on PATH so tools like ninja can be found
        env = os.environ.copy()
        venv_bin = os.path.dirname(self.vllm_env)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        result = subprocess.Popen(
            [
                self.vllm_env,
                run_file_path,
                model,
                self.config.model_dump_json(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        stdout, stderr = result.communicate()
        logger.info(stdout)

        error_msg = f"ERROR: vLLM failed with exit code {result.returncode}: {stderr}"
        assert result.returncode == 0, error_msg

        return json.loads(stdout)

    def _save_compressed_model(self, oneshot_model, processor):
        oneshot_model.save_pretrained(self.config.save_dir)
        processor.save_pretrained(self.config.save_dir)

    def _validate_recovery(self, base_results, compressed_results):
        """Validate using recovery testing - compare against base model."""
        base_metrics = base_results["results"][self.config.lmeval.task]
        compressed_metrics = compressed_results["results"][self.config.lmeval.task]
        higher_is_better_map = compressed_results.get("higher_is_better", {}).get(
            self.config.lmeval.task, {}
        )

        logger.info("=" * 80)
        logger.info("RECOVERY TESTING COMPARISON")
        logger.info("=" * 80)

        # Get default threshold from config schema
        default_threshold = self.config.lmeval.model_fields[
            "recovery_threshold"
        ].default

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
            if isinstance(self.config.lmeval.recovery_threshold, dict):
                threshold = self.config.lmeval.recovery_threshold.get(
                    metric_key, default_threshold
                )
            else:
                threshold = self.config.lmeval.recovery_threshold

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

            # Check threshold - rounds to the nearest percent - 0.94567 -> 0.95
            recovery = (torch.round(torch.tensor(recovery) * 100) / 100).item()
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
        if isinstance(self.config.lmeval.recovery_threshold, dict):
            for config_metric_key in self.config.lmeval.recovery_threshold.keys():
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

        metrics: dict = results["results"][self.config.lmeval.task]
        for metric_key, expected_val in self.config.lmeval.metrics.items():
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
                .get(self.config.lmeval.task, {})
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
        if self.config.save_dir is not None and os.path.isdir(self.config.save_dir):
            shutil.rmtree(self.config.save_dir)
