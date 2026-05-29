import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import pandas as pd
import pytest
import torch
import transformers as _transformers
import yaml
from datasets import Dataset
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from transformers import ProcessorMixin

DISABLE_LMEVAL_CACHE = os.environ.get("DISABLE_LMEVAL_CACHE", "").lower() in (
    "1",
    "true",
    "yes",
)
LMEVAL_CACHE_DIR = Path(os.environ.get("LMEVAL_CACHE_DIR", ".lmeval_cache"))
LMEVAL_CACHE_FILE = LMEVAL_CACHE_DIR / "cache.csv"


class BaseTestConfig(BaseModel):
    """
    Shared configuration for lm-eval and e2e test cases, loaded from a YAML config file.

    This configuration class serves as the foundation for defining test scenarios across
    different quantization workflows. It provides a unified interface for model,
    quantization schemes, calibration datasets, and test infrastructure requirements.

    Required fields
    ---------------
    cadence : str
        When this test runs. One of: "commit", "nightly", "weekly".
        Determines the CI cadence for this test configuration.
    model : str
        HuggingFace model ID to quantize (e.g. "meta-llama/Meta-Llama-3-8B-Instruct").
        Must be a valid model identifier on HuggingFace Hub or a local path.

    Quantization source (at least one required)
    -------------------------------------------
    scheme : str | None
        Preset quantization scheme passed directly to the modifier
        (e.g. "FP8", "FP8_DYNAMIC", "W4A16", "INT8_dyn_per_token", "NVFP4").
        Used when no recipe is provided. Ignored when entrypoint is "convert_checkpoint"
    recipe : str | None
        Path to a YAML recipe file. When both scheme and recipe are used, recipe wins.
        Required when entrypoint is "convert_checkpoint". Can be a relative path from
        the config file location or an absolute path.
    entrypoint : "oneshot" | "model_free_ptq" | "convert_checkpoint"
        Determines which quantization pathway to use (default: "oneshot"):
          - "oneshot"             : Standard quantization using all available fields
          - "model_free_ptq"      : Model-free PTQ (requires scheme)
          - "convert_checkpoint"  : Convert using pre-baked recipe (requires recipe)
    ignore : list[str] | None
        Set of layer names to ignore during model_free_ptq. Supports regex patterns.
        Only applicable when entrypoint is "model_free_ptq".

    Optional calibration dataset fields
    ------------------------------------
    dataset_id : str | None
        HuggingFace dataset ID for calibration. Leave unset to skip calibration.
        Datasets with special data-collator handling in run_oneshot_for_e2e_testing:
          - "HuggingFaceH4/ultrachat_200k"  → text, DefaultDataCollator
          - "neuralmagic/calibration"        → multimodal; set dataset_config="LLM"
          - any ID containing "flickr30k"   → multimodal, flickr30k collator
        Any other dataset ID uses DefaultDataCollator.
    dataset_config : str | None
        Dataset config/subset name (e.g. "LLM" for "neuralmagic/calibration").
        Required for datasets with multiple configurations.
    dataset_split : str | None
        Dataset split string (e.g. "train_sft", "train[:512]").
        Supports HuggingFace slice notation for limiting dataset size.
    num_calibration_samples : int
        How many samples to use for calibration (default: 512).
        Actual samples used may be less if dataset is smaller.

    Optional quantization overrides
    --------------------------------
    model_class : str
        Transformers class used to load the model (default: "AutoModelForCausalLM").
        Use e.g. "Qwen3VLForConditionalGeneration" for vision-language models.
        Must be a valid class from the transformers library.
    quant_type : "GPTQ" | "RTN" | None
        Modifier to use when no recipe is provided.
          - None / "RTN" → QuantizationModifier (default for most schemes)
          - "GPTQ"       → GPTQModifier (activation-order / GPTQ-style quantization)
        Ignored when a recipe is provided.
    max_memory : dict[int | str, int] | None
        Max memory per device for model loading. Keys can be device indices (int)
        or device names (str like "cpu"). Values are in MB.
        Example: {0: 40000, 1: 40000, "cpu": 10000}.
        Passed to from_pretrained's max_memory parameter, used for disk offloading.
    seed : int
        Random seed for reproducibility (default: 42).
        Affects dataset shuffling and quantization randomness.

    Save / output
    -------------
    save_dir : str | None
        Where to write the compressed model. Defaults to the config file's stem
        (e.g. "fp8_dynamic_per_token" for fp8_dynamic_per_token.yaml) so that
        each config always produces a unique, predictable directory without
        depending on the scheme name. Can be a relative or absolute path.

    Test infrastructure
    -------------------
    gpu_memory_utilization : float | None
        Fraction of GPU memory for vLLM to use (default: 0.70).
        Valid range is typically 0.0-1.0. Lower values leave memory for other processes.
    num_gpus : int
        Number of GPUs required for this test (default: 1).
    pipeline_parallel : bool
        Enable pipeline parallelism for vLLM serving (default: False).
        When True, pipeline_parallel_size is set to num_gpus.
        Useful for large models that don't fit on a single GPU.
    test_group : str | None
        Optional test group tag (e.g. "rhaiis") used by CI to filter test runs.
        Allows selective test execution based on environment or requirements.

    Example YAML configurations
    ----------------------------
    Basic FP8 quantization with calibration:
        ```yaml
        cadence: commit
        model: meta-llama/Meta-Llama-3-8B-Instruct
        scheme: FP8_DYNAMIC
        dataset_id: HuggingFaceH4/ultrachat_200k
        dataset_split: train[:512]
        num_calibration_samples: 512
        ```

    Model-free PTQ with layer ignoring:
        ```yaml
        cadence: nightly
        model: meta-llama/Meta-Llama-3-8B
        scheme: W4A16
        entrypoint: model_free_ptq
        ignore: ["lm_head", "model.embed_tokens"]
        num_gpus: 2
        ```

    Recipe-based conversion:
        ```yaml
        cadence: weekly
        model: Qwen/Qwen2-VL-7B-Instruct
        recipe: recipes/qwen2_vl_fp8.yaml
        entrypoint: convert_checkpoint
        model_class: Qwen3VLForConditionalGeneration
        test_group: vision
        ```
    """

    # -------------------------------------------------------------------------
    # Required
    # -------------------------------------------------------------------------
    cadence: str = Field(..., description="'commit', 'nightly', or 'weekly'")
    model: str = Field(..., description="HuggingFace model ID to quantize")

    # -------------------------------------------------------------------------
    # Quantization source — at least one must be set (enforced below)
    # -------------------------------------------------------------------------
    scheme: Optional[str] = Field(
        None,
        description=(
            "Preset quantization scheme (e.g. FP8, FP8_DYNAMIC, W4A16, "
            "INT8_dyn_per_token, NVFP4). Used when no recipe is provided."
        ),
    )
    recipe: Optional[str] = Field(
        None,
        description=(
            "Path to a quantization recipe YAML file. "
            "Takes precedence over scheme when both are set."
            "Used by `convert_checkpoint` to target specific pre-baked recipes."
        ),
    )
    entrypoint: Literal["oneshot", "model_free_ptq", "convert_checkpoint"] = Field(
        "oneshot",
        description=(
            "Entrypoint to use to create model.\n"
            "`oneshot`:\n"
            "  default entrypoint, uses all fields when they are provided\n"
            "`model_free_ptq`:\n"
            "  requires `scheme`, all other quantization arguments are ignored\n"
            "`convert_checkpoint`:\n"
            "  requires `recipe`, all other quantization arguments are ignored\n"
        ),
    )
    ignore: Optional[list[str]] = Field(
        None,
        description=(
            "Set of layer names to ignore during model_free_ptq. Regexes allowed"
        ),
    )

    # -------------------------------------------------------------------------
    # Calibration dataset (all optional — omit to skip calibration)
    # -------------------------------------------------------------------------
    dataset_id: Optional[str] = Field(
        None,
        description=(
            "HuggingFace dataset ID. Known datasets with special collator handling:\n"
            " 'HuggingFaceH4/ultrachat_200k' — text, DefaultDataCollator\n"
            " 'neuralmagic/calibration'      — multimodal (set dataset_config='LLM')\n"
            " any ID containing 'flickr30k'  — multimodal, flickr30k collator\n"
            "Any other ID uses DefaultDataCollator."
        ),
    )
    dataset_config: Optional[str] = Field(
        None,
        description="Dataset config/subset (e.g. 'LLM' for neuralmagic/calibration)",
    )
    dataset_split: Optional[str] = Field(
        None, description="Dataset split (e.g. 'train_sft', 'train[:512]')"
    )
    num_calibration_samples: int = Field(
        512, description="Number of calibration samples"
    )

    # -------------------------------------------------------------------------
    # Model / quantization overrides
    # -------------------------------------------------------------------------
    model_class: str = Field(
        "AutoModelForCausalLM",
        description=(
            "Transformers class used to load the model. "
            "Use e.g. 'Qwen3VLForConditionalGeneration' for vision-language models."
        ),
    )
    quant_type: Literal["GPTQ", "RTN"] | None = Field(
        None,
        description=(
            "Modifier used when no recipe is provided.\n"
            "  None / 'RTN' → QuantizationModifier (default)\n"
            "  'GPTQ'       → GPTQModifier"
        ),
    )
    max_memory: Optional[dict[int | str, int]] = Field(
        None,
        description="Max memory for model loading. Used to induce disk offloading",
    )
    seed: int = Field(42, description="Random seed for reproducibility")

    # -------------------------------------------------------------------------
    # Save directory
    # -------------------------------------------------------------------------
    save_dir: Optional[str] = Field(
        None,
        description=(
            "Directory to save the compressed model. "
            "If unset, defaults to the config file's stem "
            "(e.g. 'fp8_dynamic_per_token' for fp8_dynamic_per_token.yaml)."
        ),
    )

    # -------------------------------------------------------------------------
    # Test infra
    # -------------------------------------------------------------------------
    gpu_memory_utilization: Optional[float] = Field(
        0.70,
        description="GPU memory for vLLM (e.g. 0.8). Omit to use vLLM default.",
    )
    num_gpus: int = Field(
        1,
        description=(
            "Number of GPUs required for this test. "
            "Tests are skipped if fewer are available.",
        ),
    )
    pipeline_parallel: bool = Field(
        False,
        description=(
            "Enable pipeline parallelism for vLLM serving. "
            "When True, pipeline_parallel_size is set to num_gpus."
        ),
    )
    test_group: Optional[str] = Field(
        None, description="CI test group tag (e.g. 'rhaiis') used to filter test runs"
    )
    skip_sanity_check: bool = Field(
        False,
        description="Skip the sanity check that verifies vLLM generates coherent text",
    )

    @model_validator(mode="after")
    def require_scheme_or_recipe(self) -> "BaseTestConfig":
        if not self.scheme and not self.recipe:
            raise ValueError(
                "At least one of 'scheme' or 'recipe' must be provided in the config."
            )
        return self


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


def requires_compute_capability(major: int, minor: int = 0) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on GPU compute capability.

    Usage:
    @requires_compute_capability(9, 0)  # Requires H100 or higher
    def test_something():
        pass

    :param major: required major compute capability version
    :param minor: required minor compute capability version (default 0)
    """
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA not available")

    device_capability = torch.cuda.get_device_capability(0)
    has_capability = device_capability[0] > major or (
        device_capability[0] == major and device_capability[1] >= minor
    )

    reason = (
        f"Compute capability {major}.{minor} required, "
        f"found {device_capability[0]}.{device_capability[1]}"
    )
    return pytest.mark.skipif(not has_capability, reason=reason)


def torchrun(world_size: int = 1):
    """
    Pytest decorator to run a test within parallel torchrun subprocesses.

    This decorator automatically spawns torchrun when the test is run with regular
    pytest.
    When running under torchrun (detected via TORCHELASTIC_RUN_ID env var), it simply
    runs the test. The test is responsible for its own distributed initialization.

    related to https://github.com/vllm-project/compressed-tensors/blob/main/tests/test_offload/conftest.py#L81

    Usage:
        @pytest.mark.unit
        @requires_gpu(2)
        @torchrun(world_size=2)
        def test_distributed_feature():
            # Test must handle its own distributed setup
            torch.distributed.init_process_group(...)
            ...

    :param world_size: number of ranks to spawn
    """
    import subprocess
    import sys

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # We're running in a torchrun subprocess: just run the test
            if "TORCHELASTIC_RUN_ID" in os.environ:
                return func(*args, **kwargs)

            # First time calling in the main process:
            # trigger torchrun with this function as the pytest target
            else:
                module = sys.modules.get(func.__module__)
                if module is None:
                    raise RuntimeError(
                        f"Can't find module {func.__module__} for func {func.__name__}"
                    )
                file_path = module.__file__
                if file_path is None:
                    raise RuntimeError(
                        f"Module {func.__module__} has no __file__ attribute"
                    )
                func_name = func.__name__

                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    str(world_size),
                    "--log-dir",
                    "/tmp/torchrun-logs",
                    "--tee",
                    "3",
                    "--role",
                    "torchrun",
                    "-m",
                    "pytest",
                    f"{file_path}::{func_name}",
                    "-sx",
                ]

                # If coverage is enabled (--cov in PYTEST_ADDOPTS), prevent
                # pytest-cov from loading in workers by adding --no-cov.
                # Worker coverage data is still collected via .coveragerc's
                # patch = subprocess + parallel = True.
                if "--cov" in os.environ.get("PYTEST_ADDOPTS", ""):
                    cmd.append("--no-cov")

                proc = subprocess.run(cmd)
                assert proc.returncode == 0

        return wrapper

    return decorator


requires_hf_token: callable = pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping tests requiring gated model access",
)


_TRANSFORMERS_MAJOR = int(_transformers.__version__.split(".")[0])

requires_transformers_v5: pytest.MarkDecorator = pytest.mark.skipif(
    _TRANSFORMERS_MAJOR < 5,
    reason="Requires transformers v5+",
)

requires_transformers_v4: pytest.MarkDecorator = pytest.mark.skipif(
    _TRANSFORMERS_MAJOR >= 5,
    reason="Requires transformers v4 (not compatible with v5+)",
)


def _load_yaml(config_path: str):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    return None


_VALID_CADENCES = {"commit", "weekly", "nightly"}


def _validate_test_config(config: dict) -> bool:
    cadence = config.get("cadence")
    if not cadence:
        return False
    cadences = cadence if isinstance(cadence, list) else [cadence]
    return all(c in _VALID_CADENCES for c in cadences)


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
    apply_chat_template: bool
    fewshot_as_multiturn: bool
    dtype: str
    add_bos_token: bool
    trust_remote_code: bool
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

        lmeval = test_instance.config.lmeval
        seed = test_instance.config.seed

        return cls(
            model=test_instance.config.model,
            task=lmeval.task,
            num_fewshot=lmeval.num_fewshot,
            limit=lmeval.limit,
            apply_chat_template=lmeval.apply_chat_template,
            fewshot_as_multiturn=lmeval.fewshot_as_multiturn,
            dtype=lmeval.dtype,
            add_bos_token=lmeval.add_bos_token,
            trust_remote_code=lmeval.trust_remote_code,
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
            and row["apply_chat_template"] == self.apply_chat_template
            and row["fewshot_as_multiturn"] == self.fewshot_as_multiturn
            and row["dtype"] == self.dtype
            and row["add_bos_token"] == self.add_bos_token
            and row["trust_remote_code"] == self.trust_remote_code
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
                "apply_chat_template": self.apply_chat_template,
                "fewshot_as_multiturn": self.fewshot_as_multiturn,
                "dtype": self.dtype,
                "add_bos_token": self.add_bos_token,
                "trust_remote_code": self.trust_remote_code,
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
