import gc
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import yaml
from huggingface_hub import HfApi
from loguru import logger

from tests.e2e.e2e_utils import (
    run_convert_checkpoint_for_e2e_testing,
    run_model_free_ptq_for_e2e_testing,
    run_oneshot_for_e2e_testing,
)
from tests.testing_utils import BaseTestConfig, requires_gpu

HF_MODEL_HUB_NAME = "nm-testing"

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE", "tests/e2e/configs/int8_dynamic_per_token.yaml"
)
SKIP_HF_UPLOAD = os.environ.get("SKIP_HF_UPLOAD", "")
# vllm python environment
VLLM_PYTHON_ENV = os.environ.get("VLLM_PYTHON_ENV", "same")
IS_VLLM_IMAGE = False
if VLLM_PYTHON_ENV.lower() != "same" and (not Path(VLLM_PYTHON_ENV).exists()):
    IS_VLLM_IMAGE = True
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class SanityPrompt(NamedTuple):
    prompt: str
    expected: str


SANITY_PROMPTS = [
    SanityPrompt("The capital of France is", "paris"),
    SanityPrompt("The creator of the theory of relativity was Albert", "einstein"),
    SanityPrompt("The classic game is called rock, paper,", "scissors"),
]

EXPECTED_SAVED_FILES = [
    "config.json",
    r"^model(?:-\d{5}-of-\d{5})?\.safetensors$",
    "recipe.yaml",
    "tokenizer.json",
]


# Will run each test case in its own process through run_tests_in_python.sh
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

        self.config = BaseTestConfig(**eval_config)

        available_gpus = torch.accelerator.device_count()
        if available_gpus < self.config.num_gpus:
            pytest.skip(
                f"Not enough GPUs: need {self.config.num_gpus}, got {available_gpus}"
            )

        if not self.config.save_dir:
            self.config.save_dir = Path(test_data_file).stem

        if VLLM_PYTHON_ENV.lower() != "same":
            self.vllm_env = VLLM_PYTHON_ENV
        else:
            self.vllm_env = sys.executable

        logger.info("========== RUNNING ==============")
        logger.info(self.config.save_dir)

        self.api = HfApi()

    def compress_model(self, test_data_file: str):
        self.set_up(test_data_file)
        match self.config.entrypoint:
            case "convert_checkpoint":
                logger.info("============== RUNNING CONVERT CHECKPOINT ===============")
                run_convert_checkpoint_for_e2e_testing(
                    model_stub=self.config.model,
                    save_directory=self.config.save_dir,
                    recipe=self.config.recipe,
                )

            case "model_free_ptq":
                logger.info("================ RUNNING MODEL FREE PTQ =================")
                run_model_free_ptq_for_e2e_testing(
                    model_stub=self.config.model,
                    save_directory=self.config.save_dir,
                    scheme=self.config.scheme,
                    ignore=self.config.ignore or [],
                )

            case "oneshot":
                logger.info("================ RUNNING ONESHOT ======================")
                run_oneshot_for_e2e_testing(
                    model=self.config.model,
                    model_class=self.config.model_class,
                    max_memory=self.config.max_memory,
                    num_calibration_samples=self.config.num_calibration_samples,
                    max_seq_length=self.config.max_seq_length,
                    scheme=self.config.scheme,
                    dataset_id=self.config.dataset_id,
                    dataset_config=self.config.dataset_config,
                    dataset_split=self.config.dataset_split,
                    recipe=self.config.recipe,
                    quant_type=self.config.quant_type,
                    num_gpus=self.config.num_gpus,
                    save_dir=self.config.save_dir,
                    save_compressed=True,
                )
                self._check_save_dir_has_expected_files()

    def maybe_upload_to_hub(self):
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()
        # Give GPU time to fully release memory
        time.sleep(2)

        if SKIP_HF_UPLOAD.lower() != "yes":
            logger.info("================= UPLOADING TO HUB ======================")

            stub = f"{HF_MODEL_HUB_NAME}/{self.config.save_dir}-e2e"

            self.api.create_repo(
                repo_id=stub,
                exist_ok=True,
                repo_type="model",
                private=False,
            )

            self.api.upload_folder(
                repo_id=stub,
                folder_path=self.config.save_dir,
            )

    def test_vllm(self, test_data_file: str):
        self.compress_model(test_data_file)

        self.maybe_upload_to_hub()

        # Run vLLM with saved model
        if IS_VLLM_IMAGE:
            logger.info("========== RUNNING vLLM in RHAIIS vllm image ==========")
        elif VLLM_PYTHON_ENV.lower() == "same":
            logger.info("========== RUNNING vLLM in the same python env ==========")
        else:
            logger.info("========== RUNNING vLLM in a separate python env ==========")

        self._run_vllm(logger)

        self.tear_down()

    def tear_down(self):
        if self.config.save_dir is not None and os.path.isdir(self.config.save_dir):
            shutil.rmtree(self.config.save_dir)

    def _run_vllm(self, logger):
        import json
        import subprocess

        llm_kwargs = {"model": self.config.save_dir}

        # if FP8A16 scheme, must set VLLM_TEST_FORCE_FP8_MARLIN=1
        # to force usage of marlin kernel
        if self.config.scheme and "FP8A16" in self.config.scheme.upper():
            os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
        else:
            os.environ.pop("VLLM_TEST_FORCE_FP8_MARLIN", None)

        llm_kwargs["gpu_memory_utilization"] = self.config.gpu_memory_utilization

        json_scheme = json.dumps(self.config.scheme)
        json_llm_kwargs = json.dumps(llm_kwargs)
        prompts = [sp.prompt for sp in SANITY_PROMPTS]
        json_prompts = json.dumps(prompts)
        json_sampling_params = json.dumps(self.config.sampling_params)

        test_file_dir = os.path.dirname(os.path.abspath(__file__))

        if IS_VLLM_IMAGE:
            # generate python command to run in the vllm image
            RUN_SAVE_DIR = os.path.dirname(self.config.save_dir)
            run_file_path = os.path.join(RUN_SAVE_DIR, "run_vllm.py")
            shutil.copy(
                os.path.join(test_file_dir, "run_vllm.py"),
                os.path.join(RUN_SAVE_DIR, "run_vllm.py"),
            )
            cmds = [
                "python",
                run_file_path,
                f"'{json_scheme}'",
                f"'{json_llm_kwargs}'",
                f"'{json_prompts}'",
                f"'{json_sampling_params}'",
            ]
            vllm_cmd = " ".join(cmds)
            vllm_bash = os.path.join(RUN_SAVE_DIR, "run-vllm.bash")
            with open(vllm_bash, "w") as cf:
                cf.write(
                    f"""#!/bin/bash
                    export HF_HUB_OFFLINE=0
                    export VLLM_NO_USAGE_STATS=1
                    {vllm_cmd}
                    """
                )
            os.chmod(vllm_bash, 0o755)
            logger.info(f"Wrote vllm cmd into {vllm_bash}:")
            logger.info("vllm image. Run vllm cmd with kubectl.")
            result = subprocess.Popen(
                [
                    "kubectl",
                    "exec",
                    "-it",
                    VLLM_PYTHON_ENV,
                    "-n",
                    "arc-runners",
                    "--",
                    "/bin/bash",
                    vllm_bash,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            run_file_path = os.path.join(test_file_dir, "run_vllm.py")
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
                    json_scheme,
                    json_llm_kwargs,
                    json_prompts,
                    json_sampling_params,
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

        if self.config.skip_sanity_check:
            logger.info("Skipping sanity check (skip_sanity_check=true)")
            return

        stdout_lower = stdout.lower()
        for sp in SANITY_PROMPTS:
            assert (
                sp.expected in stdout_lower
            ), f"Expected '{sp.expected}' in generated output:\n{stdout}"

    def _check_save_dir_has_expected_files(self):
        files = os.listdir(self.config.save_dir)
        logger.debug(f"Saved files: {files}")

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
