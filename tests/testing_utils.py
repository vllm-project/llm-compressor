import dataclasses
import enum
import logging
import os
import unittest
from subprocess import PIPE, STDOUT, run
from typing import List, Optional, Union

import yaml
from datasets import Dataset
from transformers import AutoTokenizer

from tests.data import CustomTestConfig, TestConfig


# TODO: probably makes sense to move this type of function to a more central place,
# which can be used by __init__.py as well
def is_torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def is_gpu_available():
    """
    Check for GPU and warn if not found
    """
    try:
        import torch  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False


def requires_torch(test_case):
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def requires_gpu(test_case):
    return unittest.skipUnless(is_gpu_available(), "test requires GPU")(test_case)


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
def parse_params(
    configs_directory: Union[list, str], type: Optional[str] = None
) -> List[Union[dict, CustomTestConfig]]:
    # parses the config files provided

    config_dicts = []

    def _parse_configs_dir(current_config_dir):
        assert os.path.isdir(
            current_config_dir
        ), f"Config_directory {current_config_dir} is not a directory"

        for file in os.listdir(current_config_dir):
            config = _load_yaml(os.path.join(current_config_dir, file))
            if not config:
                continue

            cadence = os.environ.get("CADENCE", "commit")
            expected_cadence = config.get("cadence")

            if not isinstance(expected_cadence, list):
                expected_cadence = [expected_cadence]
            if cadence in expected_cadence:
                if type == "custom":
                    config = CustomTestConfig(**config)
                else:
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


def run_cli_command(cmd: List[str]):
    """
    Run a cli command and return the response. The cli command is launched through a new
    subprocess.

    :param cmd: cli command provided as a list of arguments where each argument
        should be a string
    """
    return run(cmd, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8")


def preprocess_tokenize_dataset(
    ds: Dataset, tokenizer: AutoTokenizer, max_seq_length: int
) -> Dataset:
    """
    Helper function to preprocess and tokenize a dataset according to presets

    :param ds: language dataset to preprocess and tokenize
    :param tokenizer: tokenizer to be used for tokenization
    :param max_seq_length: maximum sequence length of samples
    """
    if ds.info.dataset_name == "gsm8k":

        def preprocess(example):
            return example

        def tokenize(sample):
            return tokenizer(
                sample["question"],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )
    elif ds.info.dataset_name == "ultrachat_200k":

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )
    else:
        raise NotImplementedError(f"Cannot preprocess dataset {ds.info.dataset_name}")

    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds
