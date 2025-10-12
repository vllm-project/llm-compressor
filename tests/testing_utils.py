import dataclasses
import enum
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import Callable, List, Optional, Union

import pytest
import torch
import yaml
from datasets import Dataset
from transformers import ProcessorMixin

TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", None)


# TODO: maybe test type as decorators?
class TestType(Enum):
    SANITY = "sanity"
    REGRESSION = "regression"
    SMOKE = "smoke"


class Cadence(Enum):
    COMMIT = "commit"
    WEEKLY = "weekly"
    NIGHTLY = "nightly"


@dataclass
class TestConfig:
    test_type: TestType
    cadence: Cadence


def _enough_gpus(num_required_gpus):
    try:
        import torch  # noqa: F401

        return torch.cuda.device_count() >= num_required_gpus
    except ImportError:
        return False


def requires_gpu(test_case_or_num):
    """
    Pytest decorator to skip based on number of available GPUs.

    Designed for backwards compatibility with the old requires_gpu decorator
    Usage:
    @requires_gpu
    def test_something():
        # only runs if there is at least 1 GPU available
        pass

    @requires_gpu(2)
    def test_something_else():
        # only runs if there are at least 2 GPUs available
        pass
    """
    if isinstance(test_case_or_num, int):
        num_required_gpus = test_case_or_num
    else:
        num_required_gpus = 1

    decorator = pytest.mark.skipif(
        not _enough_gpus(num_required_gpus),
        reason=f"Not enough GPUs available, {num_required_gpus} GPUs required",
    )
    if isinstance(test_case_or_num, int):
        return decorator
    else:
        return decorator(test_case_or_num)


def requires_gpu_mem(required_amount: Union[int, float]) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on total available GPU memory (across all GPUs). This
    plays nicely with the CUDA_VISIBLE_DEVICES environment variable.

    Note: make sure to account for measured memory vs. simple specs. For example, H100
    has '80 GiB' VRAM, however, the actual number, at least per PyTorch, is ~79.2 GiB.

    :param amount: amount of required GPU memory in GiB
    """

    vram_bytes = sum(
        torch.cuda.mem_get_info(device_id)[1]
        for device_id in range(torch.cuda.device_count())
    )
    actual_vram = vram_bytes / 1024**3
    reason = (
        f"{required_amount} GiB GPU memory required, "
        f"{actual_vram:.1f} GiB GPU memory found"
    )
    return pytest.mark.skipif(required_amount > actual_vram, reason=reason)


def _load_yaml(config_path: str):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    return None


def _validate_test_config(config: dict):
    for f in dataclasses.fields(TestConfig):
        if f.name not in config:
            return False
        config_value = config.get(f.name)
        if issubclass(f.type, enum.Enum):
            try:
                f.type(config_value)
            except ValueError:
                return False
    return True


# Set cadence in the config. The environment must set if nightly, weekly or commit
# tests are running
def parse_params(configs_directory: Union[list, str]) -> List[dict]:
    # parses the config files provided

    config_dicts = []

    def _parse_configs_dir(current_config_dir):
        assert os.path.isdir(
            current_config_dir
        ), f"Config_directory {current_config_dir} is not a directory"

        for file in os.listdir(current_config_dir):
            config_path = os.path.join(current_config_dir, file)
            if TEST_DATA_FILE is not None:
                if not config_path.endswith(TEST_DATA_FILE):
                    continue

            config = _load_yaml(config_path)
            if not config:
                continue

            config["testconfig_path"] = config_path

            cadence = os.environ.get("CADENCE", "commit")
            expected_cadence = config.get("cadence")

            if not isinstance(expected_cadence, list):
                expected_cadence = [expected_cadence]
            if cadence in expected_cadence:
                if not _validate_test_config(config):
                    raise ValueError(
                        "The config provided does not comply with the expected "
                        "structure. See tests.data.TestConfig for the expected "
                        "fields."
                    )
                config_dicts.append(config)
            else:
                logging.info(
                    f"Skipping testing model: {file} for cadence: {expected_cadence}"
                )

    if isinstance(configs_directory, list):
        for config in configs_directory:
            _parse_configs_dir(config)
    else:
        _parse_configs_dir(configs_directory)

    return config_dicts


def run_cli_command(cmd: List[str], cwd: Optional[Union[str, Path]] = None):
    """
    Run a cli command and return the response. The cli command is launched through a new
    subprocess.

    :param cmd: cli command provided as a list of arguments where each argument
        should be a string
    """
    return run(cmd, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8", cwd=cwd)


def process_dataset(
    ds: Dataset, processor: ProcessorMixin, max_seq_length: int
) -> Dataset:
    """
    Helper function to preprocess and tokenize a dataset according to presets

    :param ds: language dataset to preprocess and tokenize
    :param tokenizer: tokenizer to be used for tokenization
    :param max_seq_length: maximum sequence length of samples
    """
    ds_name = ds.info.dataset_name.lower()
    if ds_name == "gsm8k":

        def process(sample):
            return processor(
                sample["question"],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == "ultrachat_200k":

        def process(sample):
            return processor(
                processor.apply_chat_template(
                    sample["messages"],
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == "llm_compression_calibration":

        def process(sample):
            return processor(
                processor.apply_chat_template(
                    sample["text"],
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == "open-platypus":
        # use the output rather than the instruction
        def process(sample):
            return processor(
                processor.apply_chat_template(
                    sample["output"],
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == "slimorca-deduped-cleaned-corrected":
        # find the first element corresponding to a message from a human
        def process(sample):
            conversation_idx = 0
            for idx, conversation in enumerate(sample["conversations"]):
                if conversation["from"] == "human":
                    conversation_idx = idx
                    break
            return processor(
                processor.apply_chat_template(
                    sample["conversations"][conversation_idx]["value"],
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == "flickr30k":

        def process(sample):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What does the image show?"},
                    ],
                }
            ]
            return {
                "text": processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                ),
                "images": sample["image"],
            }

    else:
        raise NotImplementedError(f"Cannot preprocess dataset {ds.info.dataset_name}")

    ds = ds.map(process, remove_columns=ds.column_names)

    return ds


def requires_cadence(cadence: Union[str, List[str]]) -> Callable:
    cadence = [cadence] if isinstance(cadence, str) else cadence
    current_cadence = os.environ.get("CADENCE", "commit")

    return pytest.mark.skipif(
        (current_cadence not in cadence), reason="cadence mismatch"
    )
