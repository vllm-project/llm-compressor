import dataclasses
import enum
import logging
import os
import unittest
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import List, Optional, Union

import yaml
from datasets import Dataset
from transformers import ProcessorMixin

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
            config_path = os.path.join(current_config_dir, file)
            config = _load_yaml(config_path)
            if not config:
                continue

            config["testconfig_path"] = config_path

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

    elif ds_name == "pile-val-backup":

        def preprocess(example):
            return {
                "input_ids": processor.encode(example["text"].strip())[:max_seq_length]
            }

        ds = ds.map(preprocess, remove_columns=ds.column_names)
        # Note: potentially swap filtering to pad for AWQ
        ds = ds.filter(lambda example: len(example["input_ids"]) >= max_seq_length)
        return ds

    else:
        raise NotImplementedError(f"Cannot preprocess dataset {ds.info.dataset_name}")

    ds = ds.map(process, remove_columns=ds.column_names)

    return ds
