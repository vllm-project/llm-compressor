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


def is_quay_image(url: str) -> bool:
    pattern = r"^quay\.io/[a-z0-9][a-z0-9-_]*/[a-z0-9][a-z0-9-_/]*:[\w][\w.-]*$"
    return re.match(pattern, url) is not None

HF_MODEL_HUB_NAME = "nm-testing"

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE", "tests/e2e/vLLM/configs/int8_dynamic_per_token.yaml"
)
SKIP_HF_UPLOAD = os.environ.get("SKIP_HF_UPLOAD", "")
# vllm environment: image url, deployed runner name, same (default), or the path of vllm virtualenv
VLLM_PYTHON_ENV = os.environ.get("VLLM_PYTHON_ENV", "same")
IS_VLLM_IMAGE = False
IS_VLLM_IMAGE_DEPLOYED=False
RUN_SAVE_DIR=os.environ.get("RUN_SAVE_DIR", "none")
VLLM_VOLUME_MOUNT_DIR=os.environ.get("VLLM_VOLUME_MOUNT_DIR", "/opt/app-root/runs")
# when using vllm image, needs to save the generated model and vllm command
if VLLM_PYTHON_ENV.lower() != "same" and (not Path(VLLM_PYTHON_ENV).exists()):
    IS_VLLM_IMAGE = True
    if not is_quay_image(VLLM_PYTHON_ENV):
        IS_VLLM_IMAGE_DEPLOYED = True
        assert RUN_SAVE_DIR != "none", "To use vllm image must set RUN_SAVE_DIR too!"

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
        #self.is_vllm_image = IS_VLLM_IMAGE
        if VLLM_PYTHON_ENV.lower() == "same":
            self.vllm_env = sys.executable
        else:
            self.vllm_env = VLLM_PYTHON_ENV

        if RUN_SAVE_DIR != "none":
            assert Path(RUN_SAVE_DIR).exists(), f"RUN_SAVE_DIR path doesn't exist: {RUN_SAVE_DIR}"
            self.run_save_dir = RUN_SAVE_DIR
            # RUN_SAVE_DIR overwrites config save_dir
            self.save_dir = os.path.join(RUN_SAVE_DIR, self.model.split("/")[1] + f"-{self.scheme}")

        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"

        logger.info("========== RUNNING ==============")
        logger.info(f"model save dir: {self.save_dir}")

        # script to run vllm if using vllm image
        if IS_VLLM_IMAGE:
            self.vllm_bash = os.path.join(RUN_SAVE_DIR, "run-vllm.bash")
            logger.info(f"vllm bash save dir: {self.vllm_bash}")

        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]
        self.api = HfApi()

    def test_vllm(self, test_data_file: str):
        # Run vLLM with saved model

        self.set_up(test_data_file)
        # not need this anymore?
        #if not self.save_dir:
        #    self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
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

        # if vllm image is used, don't upload
        if SKIP_HF_UPLOAD.lower() != "yes" and not IS_VLLM_IMAGE:
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

        if IS_VLLM_IMAGE:
            logger.info("========== To run vLLM with vllm image ==========")
        else:
            if VLLM_PYTHON_ENV.lower() == "same":
                logger.info("========== RUNNING vLLM in the same python env ==========")
            else:
                logger.info("========== RUNNING vLLM in a separate python env ==========")

        self._run_vllm(logger)

        self.tear_down()

    def tear_down(self):
        # model save_dir is needed for vllm image testing
        if not IS_VLLM_IMAGE and self.save_dir is not None and os.path.isdir(self.save_dir):
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
        if IS_VLLM_IMAGE:
            llm_kwargs = {"model":
                self.save_dir.replace(RUN_SAVE_DIR, VLLM_VOLUME_MOUNT_DIR)}

        if self.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization

        json_scheme = json.dumps(self.scheme)
        json_llm_kwargs = json.dumps(llm_kwargs)
        json_prompts = json.dumps(self.prompts)

        test_file_dir = os.path.dirname(os.path.abspath(__file__))

        logger.info("Run vllm using env:")
        logger.info(self.vllm_env)

        if IS_VLLM_IMAGE:
            run_file_path = os.path.join(VLLM_VOLUME_MOUNT_DIR, "run_vllm.py")
            shutil.copy(os.path.join(test_file_dir, "run_vllm.py"), 
                os.path.join(RUN_SAVE_DIR, "run_vllm.py"))
            cmds = ["python", run_file_path, f"'{json_scheme}'",
                    f"'{json_llm_kwargs}'", f"'{json_prompts}'"]
            vllm_cmd = " ".join(cmds)
            with open(self.vllm_bash, "w") as cf:
                cf.write(f"""#!/bin/bash
                    export HF_HUB_OFFLINE=0
                    export VLLM_NO_USAGE_STATS=1
                    {vllm_cmd}
                    """)
            os.chmod(self.vllm_bash, 0o755)
            logger.info(f"Wrote vllm cmd into {self.vllm_bash}:")
            logger.info(vllm_cmd)
            if IS_VLLM_IMAGE_DEPLOYED:
                logger.info("vllm image is deployed. Run vllm cmd with kubectl.")
                cmds = [f"kubectl exec -it {VLLM_PYTHON_ENV} -n arc-runners",
                        f"-- /bin/bash {RUN_SAVE_DIR}/run-vllm.bash"]
                kubectl_cmd = " ".join(cmds)
                logger.info(f"kubectl command: {kubectl_cmd}")
                result = subprocess.Popen(
                    [
                     "kubectl", "exec", "-it",
                     VLLM_PYTHON_ENV, "-n", "arc-runners",
                     "--", "/bin/bash", f"{RUN_SAVE_DIR}/run-vllm.bash",
                    ],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True)
            else:
                cmds = ["podman run --rm --device nvidia.com/gpu=all --entrypoint",
                    self.vllm_bash.replace(RUN_SAVE_DIR, VLLM_VOLUME_MOUNT_DIR),
                    "-v", f"{RUN_SAVE_DIR}:{VLLM_VOLUME_MOUNT_DIR}",
                    VLLM_PYTHON_ENV]
                podman_cmd = " ".join(cmds)
                logger.info(f"podman command: {podman_cmd}")
                result = subprocess.Popen(
                    [
                     "podman", "run", "--rm",
                     "--device", "nvidia.com/gpu=all", "--entrypoint",
                     self.vllm_bash.replace(RUN_SAVE_DIR, VLLM_VOLUME_MOUNT_DIR),
                     "-v", f"{RUN_SAVE_DIR}:{VLLM_VOLUME_MOUNT_DIR}",
                     VLLM_PYTHON_ENV,
                   ],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True)
        else:
            run_file_path = os.path.join(test_file_dir, "run_vllm.py")
            logger.info("Run vllm in subprocess.Popen using python env:")
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
