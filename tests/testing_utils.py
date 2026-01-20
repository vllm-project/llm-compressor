import dataclasses
import enum
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pytest
import torch
import yaml
from datasets import Dataset
from loguru import logger
from transformers import ProcessorMixin

TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", None)
DISABLE_LMEVAL_CACHE = os.environ.get("DISABLE_LMEVAL_CACHE", "").lower() in (
    "1",
    "true",
    "yes",
)
LMEVAL_CACHE_DIR = Path(os.environ.get("LMEVAL_CACHE_DIR", ".lmeval_cache"))
LMEVAL_CACHE_FILE = LMEVAL_CACHE_DIR / "cache.csv"


def _sha256_hash(text: str, length: Optional[int] = None) -> str:
    hash_result = hashlib.sha256(text.encode()).hexdigest()
    return hash_result[:length] if length else hash_result


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


requires_hf_token: callable = pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping tests requiring gated model access",
)


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

    # "neuralmagic/calibration"
    elif ds_name == "calibration":

        def process(example):
            messages = []
            for message in example["messages"]:
                messages.append(
                    {
                        "role": message["role"],
                        "content": [{"type": "text", "text": message["content"]}],
                    }
                )

            return processor.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                tokenize=True,
                add_special_tokens=False,
                return_dict=True,
                add_generation_prompt=False,
            )

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


@dataclass(frozen=True)
class LMEvalCacheKey:
    """Cache key for LM Eval results based on evaluation parameters."""

    model: str
    task: str
    num_fewshot: int
    limit: int
    batch_size: int
    model_args_hash: str
    lmeval_version: str
    seed: Optional[int]

    @classmethod
    def from_test_instance(cls, test_instance: Any) -> "LMEvalCacheKey":
        """Create cache key from test instance."""
        try:
            import lm_eval

            lmeval_version = lm_eval.__version__
        except (ImportError, AttributeError):
            lmeval_version = "unknown"

        lmeval = test_instance.lmeval
        model_args_json = json.dumps(lmeval.model_args, sort_keys=True)
        seed = getattr(test_instance, "seed", None)

        return cls(
            model=test_instance.model,
            task=lmeval.task,
            num_fewshot=lmeval.num_fewshot,
            limit=lmeval.limit,
            batch_size=lmeval.batch_size,
            model_args_hash=_sha256_hash(model_args_json, 16),
            lmeval_version=lmeval_version,
            seed=seed,
        )

    def _matches(self, row: pd.Series) -> bool:
        """Check if a DataFrame row matches this cache key."""
        # Handle NaN for seed comparison (pandas reads None as NaN)
        seed_matches = (pd.isna(row["seed"]) and self.seed is None) or (
            row["seed"] == self.seed
        )
        return (
            row["model"] == self.model
            and row["task"] == self.task
            and row["num_fewshot"] == self.num_fewshot
            and row["limit"] == self.limit
            and row["batch_size"] == self.batch_size
            and row["model_args_hash"] == self.model_args_hash
            and row["lmeval_version"] == self.lmeval_version
            and seed_matches
        )

    def get_cached_result(self) -> Optional[Dict]:
        """Load cached result from CSV file."""
        if not LMEVAL_CACHE_FILE.exists():
            return None

        try:
            df = pd.read_csv(LMEVAL_CACHE_FILE)
            matches = df[df.apply(self._matches, axis=1)]

            if matches.empty:
                return None

            return json.loads(matches.iloc[0]["result"])

        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
            return None

    def store_result(self, result: Dict) -> None:
        """Store result in CSV file."""
        try:
            LMEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            new_row = {
                "model": self.model,
                "task": self.task,
                "num_fewshot": self.num_fewshot,
                "limit": self.limit,
                "batch_size": self.batch_size,
                "model_args_hash": self.model_args_hash,
                "lmeval_version": self.lmeval_version,
                "seed": self.seed,
                "result": json.dumps(result, default=str),
            }

            # Load existing cache or create new
            if LMEVAL_CACHE_FILE.exists():
                df = pd.read_csv(LMEVAL_CACHE_FILE)
                # Remove duplicate entries for this key
                df = df[~df.apply(self._matches, axis=1)]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])

            df.to_csv(LMEVAL_CACHE_FILE, index=False)
            logger.info(f"LM-Eval cache WRITE: {self.model}/{self.task}")

        except Exception as e:
            logger.debug(f"Cache write failed: {e}")


def cached_lm_eval_run(func: Callable) -> Callable:
    """
    Decorator to cache lm_eval results in CSV format.

    Caches results based on model, task, num_fewshot, limit, batch_size,
    and model_args to avoid redundant base model evaluations.

    Environment variables:
        DISABLE_LMEVAL_CACHE: Set to "1"/"true"/"yes" to disable
        LMEVAL_CACHE_DIR: Custom cache directory (default: .lmeval_cache)
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip caching if disabled
        if DISABLE_LMEVAL_CACHE:
            return func(self, *args, **kwargs)

        # Try to get cached result
        cache_key = LMEvalCacheKey.from_test_instance(self)
        if (cached_result := cache_key.get_cached_result()) is not None:
            logger.info(f"LM-Eval cache HIT: {cache_key.model}/{cache_key.task}")
            return cached_result

        # Run evaluation and cache result
        logger.info(f"LM-Eval cache MISS: {cache_key.model}/{cache_key.task}")
        result = func(self, *args, **kwargs)
        cache_key.store_result(result)

        return result

    return wrapper
