"""
Helper variables and functions for integrating LLM Compressor with
huggingface/transformers flows
"""

import os
from typing import TYPE_CHECKING, Optional

from loguru import logger
from transformers.trainer_utils import get_last_checkpoint

if TYPE_CHECKING:
    from llmcompressor.transformers import ModelArguments, TrainingArguments

if TYPE_CHECKING:
    from llmcompressor.transformers import ModelArguments, TrainingArguments


__all__ = [
    "DEFAULT_RECIPE_NAME",
    "detect_last_checkpoint",
]

DEFAULT_RECIPE_NAME = "recipe.yaml"


def detect_last_checkpoint(
    training_args: "TrainingArguments",  # noqa 821
    model_args: Optional["ModelArguments"] = None,  # noqa 821
):
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if training_args.run_stages and model_args is not None:
            model = (
                model_args.model
                if hasattr(model_args, "model")
                else model_args.model_name_or_path
            )
            if os.path.isdir(model):
                last_checkpoint = get_last_checkpoint(model_args.model_name_or_path)
        if last_checkpoint is None and (len(os.listdir(training_args.output_dir)) > 0):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already "
                "exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change  the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint
