"""
Dataset argument classes for LLM compression workflows.

This module defines dataclass-based argument containers for configuring dataset
loading, preprocessing, and calibration parameters across different dataset
sources and processing pipelines. Supports various input formats including
HuggingFace datasets, custom JSON/CSV files, and DVC-managed datasets.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DVCDatasetArguments:
    """
    Arguments for calibration using DVC
    """

    dvc_data_repository: str | None = field(
        default=None,
        metadata={"help": "Path to repository used for dvc_dataset_path"},
    )


@dataclass
class CustomDatasetArguments(DVCDatasetArguments):
    """
    Arguments for calibration using custom datasets
    """

    dataset_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the custom dataset. Supports json, csv, dvc. "
                "For DVC, the to dvc dataset to load, of format dvc://path. "
                "For csv or json, the path containing the dataset. "
            ),
        },
    )

    text_column: str = field(
        default="text",
        metadata={
            "help": (
                "Optional key to be used as the `text` input to tokenizer/processor "
                "after dataset preprocesssing"
            )
        },
    )

    remove_columns: None | str | list[str] = field(
        default=None,
        metadata={"help": "Column names to remove after preprocessing (deprecated)"},
    )

    preprocessing_func: None | str | Callable = field(
        default=None,
        metadata={
            "help": (
                "Typically a function which applies a chat template. Can take the form "
                "of either a function to apply to the dataset, a name defined in "
                "src/llmcompressor/transformers/utils/preprocessing_functions.py, or "
                "a path to a function definition of the form /path/to/file.py:func"
            )
        },
    )

    batch_size: int = field(
        default=1,
        metadata={
            "help": (
                "Calibration batch size. During calibration, LLM Compressor disables "
                "lm_head output computations to reduce memory usage from large "
                "batch sizes. Large batch sizes may result in excess padding or "
                "truncation, depending on the data_collator"
            )
        },
    )

    data_collator: str | Callable = field(
        default="truncation",
        metadata={
            "help": (
                "The function to use to form a batch from the dataset. Can also "
                "specify 'truncation' or 'padding' to truncate or pad non-uniform "
                "sequence lengths in a batch. Defaults to 'truncation'."
            )
        },
    )


@dataclass
class DatasetArguments(CustomDatasetArguments):
    """
    Arguments pertaining to what data we are going to use for
    calibration

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    dataset: str | None = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset to use (via the datasets library). "
                "Supports input as a string or DatasetDict from HF"
            )
        },
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={
            "help": ("The configuration name of the dataset to use"),
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer  than this will be truncated, sequences shorter will "
            "be padded."
        },
    )
    concatenate_data: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to concatenate datapoints to fill max_seq_length"
        },
    )
    raw_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional keyboard args to pass to datasets load_data"},
    )
    splits: None | str | list[str] | dict[str, str] = field(
        default=None,
        metadata={"help": "Optional percentages of each split to download"},
    )
    num_calibration_samples: int | None = field(
        default=512,
        metadata={"help": "Number of samples to use for one-shot calibration"},
    )
    shuffle_calibration_samples: bool = field(
        default=True,
        metadata={
            "help": "whether to shuffle the dataset before selecting calibration data"
        },
    )
    streaming: bool | None = field(
        default=False,
        metadata={"help": "True to stream data from a cloud dataset"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of workers to use for dataset processing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    min_tokens_per_module: float | None = field(
        default=None,
        metadata={
            "help": (
                "The minimum percentage of tokens (out of the total number) "
                "that the module should 'receive' throughout the forward "
                "pass of the calibration. If a module receives fewer tokens, "
                "a warning will be logged. Defaults to 1/num_of_experts."
                "note: this argument is only relevant for MoE models"
            ),
        },
    )
    moe_calibrate_all_experts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to calibrate all experts during MoE model calibration. "
                "When True, all experts will see all tokens during calibration, "
                "ensuring proper quantization statistics for all experts. "
                "When False, only routed experts will be used. "
                "Only relevant for MoE models. Default is True."
            ),
        },
    )
    # --- pipeline arguments --- #
    pipeline: str | None = field(
        default="independent",
        metadata={
            "help": "Calibration pipeline used to calibrate model"
            "Options: ['basic', 'datafree', 'sequential', independent]"
        },
    )
    tracing_ignore: list[str] = field(
        default_factory=lambda: [
            "_update_causal_mask",
            "create_causal_mask",
            "_update_mamba_mask",
            "make_causal_mask",
            "get_causal_mask",
            "mask_interface",
            "mask_function",
            "_prepare_4d_causal_attention_mask",
            "_prepare_fsmt_decoder_inputs",
            "_prepare_4d_causal_attention_mask_with_cache_position",
            "_update_linear_attn_mask",
            "project_per_layer_inputs",
        ],
        metadata={
            "help": "List of functions to ignore during tracing, either "
            "{module}.{method_name} or {function_name}"
        },
    )
    sequential_targets: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of layer targets for the sequential pipeline. "
            "This is typically a single DecoderLayer. "
            "Not specifying this argument will cause the sequential pipeline to "
            "default to using the `no_split_params` specified by the HF model "
            "definition"
        },
    )
    sequential_offload_device: str = field(
        default="cpu",
        metadata={
            "help": "Device used to offload intermediate activations between "
            "sequential layers. It is recommended to use `cuda:1` if using more "
            "than one gpu. Default is cpu."
        },
    )
    quantization_aware_calibration: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable quantization-aware calibration in the pipeline. "
            "When True, quantization is applied during forward pass in calibration. "
            "When False, quantization is disabled during forward pass in calibration. "
            "Default is set to True."
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of worker processes for data loading. Set to 0 to disable "
            "multiprocessing. Note: Custom data collators may not work with "
            "multiprocessing. Default is 0."
        },
    )

    def is_dataset_provided(self) -> bool:
        return self.dataset is not None or self.dataset_path is not None
