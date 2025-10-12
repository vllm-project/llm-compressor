"""
Dataset argument classes for LLM compression workflows.

This module defines dataclass-based argument containers for configuring dataset
loading, preprocessing, and calibration parameters across different dataset
sources and processing pipelines. Supports various input formats including
HuggingFace datasets, custom JSON/CSV files, and DVC-managed datasets.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import DefaultDataCollator


@dataclass
class DVCDatasetArguments:
    """
    Arguments for training using DVC
    """

    dvc_data_repository: Optional[str] = field(
        default=None,
        metadata={"help": "Path to repository used for dvc_dataset_path"},
    )


@dataclass
class CustomDatasetArguments(DVCDatasetArguments):
    """
    Arguments for training using custom datasets
    """

    dataset_path: Optional[str] = field(
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

    remove_columns: Union[None, str, List] = field(
        default=None,
        metadata={"help": "Column names to remove after preprocessing (deprecated)"},
    )

    preprocessing_func: Union[None, str, Callable] = field(
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

    data_collator: Callable[[Any], Any] = field(
        default_factory=lambda: DefaultDataCollator(),
        metadata={"help": "The function to used to form a batch from the dataset"},
    )


@dataclass
class DatasetArguments(CustomDatasetArguments):
    """
    Arguments pertaining to what data we are going to input our model for
    calibration, training

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset to use (via the datasets library). "
                "Supports input as a string or DatasetDict from HF"
            )
        },
    )
    dataset_config_name: Optional[str] = field(
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
    raw_kwargs: Dict = field(
        default_factory=dict,
        metadata={"help": "Additional keyboard args to pass to datasets load_data"},
    )
    splits: Union[None, str, List, Dict] = field(
        default=None,
        metadata={"help": "Optional percentages of each split to download"},
    )
    num_calibration_samples: Optional[int] = field(
        default=512,
        metadata={"help": "Number of samples to use for one-shot calibration"},
    )
    calibrate_moe_context: bool = field(
        default=False,
        metadata={
            "help": "If during calibration, the MoE context should be enabled "
            "for the given model. This usually involves updating all MoE modules "
            "in the model for the duration of calibration. See moe_context under "
            "modeling/prepare.py for a list of supported MoEs and their updated "
            "module definitions"
        },
    )
    shuffle_calibration_samples: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to shuffle the dataset before selecting calibration data"
        },
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "True to stream data from a cloud dataset"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    min_tokens_per_module: Optional[float] = field(
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
    # --- pipeline arguments --- #
    pipeline: Optional[str] = field(
        default="independent",
        metadata={
            "help": "Calibration pipeline used to calibrate model"
            "Options: ['basic', 'datafree', 'sequential', 'layer_sequential', "
            "independent]"
        },
    )
    tracing_ignore: List[str] = field(
        default_factory=lambda: [
            "_update_causal_mask",
            "create_causal_mask",
            "make_causal_mask",
            "get_causal_mask",
            "mask_interface",
            "mask_function",
            "_prepare_4d_causal_attention_mask",
            "_prepare_fsmt_decoder_inputs",
            "_prepare_4d_causal_attention_mask_with_cache_position",
            "_update_linear_attn_mask",
        ],
        metadata={
            "help": "List of functions to ignore during tracing, either "
            "{module}.{method_name} or {function_name}"
        },
    )
    sequential_targets: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of layer targets for the sequential pipeline. "
            "This is typically a single DecoderLayer. "
            "Not specifying this argument will cause the sequential pipeline to "
            "default to using the `no_split_params` specified by the HF model "
            "definition"
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

    def is_dataset_provided(self) -> bool:
        return self.dataset is not None or self.dataset_path is not None
