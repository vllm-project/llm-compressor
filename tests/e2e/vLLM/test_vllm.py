import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from huggingface_hub import HfApi
from loguru import logger

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.test_timer.timer_utils import get_singleton_manager, log_time
from tests.testing_utils import requires_gpu

HF_MODEL_HUB_NAME = "nm-testing"

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE", "tests/e2e/vLLM/configs/int8_dynamic_per_token.yaml"
)
SKIP_HF_UPLOAD = os.environ.get("SKIP_HF_UPLOAD", "")
# vllm python environment
VLLM_PYTHON_ENV = os.environ.get("VLLM_PYTHON_ENV", "same")
TIMINGS_DIR = os.environ.get("TIMINGS_DIR", "timings/e2e-test_vllm")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
EXPECTED_SAVED_FILES = [
    "config.json",
    r"^model(?:-\d{5}-of-\d{5})?\.safetensors$",
    "recipe.yaml",
    "tokenizer.json",
]


# Will run each test case in its own process through run_tests.sh
# emulating vLLM CI testing
@requires_gpu(1)
@pytest.mark.parametrize(
    "test_data_file", [pytest.param(TEST_DATA_FILE, id=TEST_DATA_FILE)]
)
class TestvLLM:
    """
    The following test quantizes a model using a preset scheme or recipe,
    runs the model using vLLM, and then pushes the model to the hub for
    future use. Each test case is focused on a specific quantization type
    (e.g W4A16 with grouped quantization, W4N16 with channel quantization).
    To add a new test case, a new config has to be added to the `configs` folder.
    The tests run on a cadence defined by the `cadence` field. Each config defines
    the model to quantize. Optionally, a dataset id and split can be provided for
    calibration. Finally, all config files must list a scheme. The scheme can be a
    preset scheme from
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
        self.scheme = eval_config.get("scheme")
        self.dataset_id = eval_config.get("dataset_id")
        self.dataset_config = eval_config.get("dataset_config")
        self.dataset_split = eval_config.get("dataset_split")
        self.recipe = eval_config.get("recipe")
        self.quant_type = eval_config.get("quant_type")
        self.save_dir = eval_config.get("save_dir")
        self.save_compressed = eval_config.get("save_compressed", True)
        self.num_calibration_samples = eval_config.get("num_calibration_samples", 256)
        self.max_seq_length = eval_config.get("max_seq_length", 2048)
        # GPU memory utilization - only set if explicitly provided in config
        self.gpu_memory_utilization = eval_config.get("gpu_memory_utilization")
        # vllm python env - if same, use the current python env, otherwise use
        # the python passed in VLLM_PYTHON_ENV
        if VLLM_PYTHON_ENV.lower() != "same":
            self.vllm_env = VLLM_PYTHON_ENV
        else:
            self.vllm_env = sys.executable

        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"

        logger.info("========== RUNNING ==============")
        logger.info(self.save_dir)

        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]
        self.api = HfApi()

    def test_vllm(self, test_data_file: str):
        # Run vLLM with saved model

        self.set_up(test_data_file)
        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
        oneshot_model, tokenizer = run_oneshot_for_e2e_testing(
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

        # check that session contains recipe
        self._check_session_contains_recipe()

        logger.info("================= SAVING TO DISK ======================")
        self._save_compressed_model(oneshot_model=oneshot_model, tokenizer=tokenizer)

        recipe_path = os.path.join(self.save_dir, "recipe.yaml")

        # check that expected files exist
        self._check_save_dir_has_expected_files()

        # Use the session to fetch the recipe;
        # Reset session for next test case
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)
        session.reset()

        if SKIP_HF_UPLOAD.lower() != "yes":
            logger.info("================= UPLOADING TO HUB ======================")

            stub = f"{HF_MODEL_HUB_NAME}/{self.save_dir}-e2e"

            self.api.create_repo(
                repo_id=stub,
                exist_ok=True,
                repo_type="model",
                private=False,
            )

            self.api.upload_folder(
                repo_id=stub,
                folder_path=self.save_dir,
            )

        if VLLM_PYTHON_ENV.lower() == "same":
            logger.info("========== RUNNING vLLM in the same python env ==========")
        else:
            logger.info("========== RUNNING vLLM in a separate python env ==========")

        self._run_vllm(logger)

        self.tear_down()

    def tear_down(self):
        if self.save_dir is not None and os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)

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

    @log_time
    def _save_compressed_model(self, oneshot_model, tokenizer):
        oneshot_model.save_pretrained(
            self.save_dir, save_compressed=self.save_compressed
        )
        tokenizer.save_pretrained(self.save_dir)

    @log_time
    def _run_vllm(self, logger):
        import json
        import subprocess

        llm_kwargs = {"model": self.save_dir}

        if self.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization

        json_scheme = json.dumps(self.scheme)
        json_llm_kwargs = json.dumps(llm_kwargs)
        json_prompts = json.dumps(self.prompts)

        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(test_file_dir, "run_vllm.py")

        logger.info("Run vllm in subprocess.Popen() using python env:")
        logger.info(self.vllm_env)

        result = subprocess.Popen(
            [self.vllm_env, run_file_path, json_scheme, json_llm_kwargs, json_prompts],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = result.communicate()
        logger.info(stdout)

        error_msg = f"ERROR: vLLM failed with exit code {result.returncode}: {stderr}"
        assert result.returncode == 0, error_msg

    def _check_session_contains_recipe(self) -> None:
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        assert recipe_yaml_str is not None

    def _check_save_dir_has_expected_files(self):
        files = os.listdir(self.save_dir)
        logger.debug("Saved files: ", files)

        matched_patterns = set()

        for expected in EXPECTED_SAVED_FILES:
            # Find all files matching the expected pattern
            matches = [
                file
                for file in files
                if (
                    re.fullmatch(expected, file)
                    if expected.startswith("^")
                    else file == expected
                )
            ]
            if len(matches) > 0:
                matched_patterns.add(expected)

        assert len(matched_patterns) == len(EXPECTED_SAVED_FILES), (
            "expected: ",
            EXPECTED_SAVED_FILES,
            "\n saved: ",
            list(matched_patterns),
        )
