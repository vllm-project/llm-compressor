"""
Helper variables and functions for integrating LLM Compressor with
huggingface/transformers flows
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import requests
from huggingface_hub import HUGGINGFACE_CO_URL_HOME, hf_hub_download
from loguru import logger
from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint

if TYPE_CHECKING:
    from llmcompressor.args import ModelArguments, TrainingArguments

__all__ = [
    "RECIPE_FILE_NAME",
    "detect_last_checkpoint",
    "is_model_ct_quantized_from_path",
]

RECIPE_FILE_NAME = "recipe.yaml"


def detect_last_checkpoint(
    training_args: "TrainingArguments",
    model_args: Optional["ModelArguments"] = None,
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


def is_model_ct_quantized_from_path(path: str) -> bool:
    """
    Determine if model from path is quantized based
    on the config

    :param path: path to the model or HF stub
    :return: True if config contains quantization_config from the given path

    """
    config = AutoConfig.from_pretrained(path)
    if config is not None:
        if (
            hasattr(config, "quantization_config")
            and config.quantization_config["quant_method"] == "compressed-tensors"
        ):
            return True
    return False


def infer_recipe_from_model_path(model_path: Union[str, Path]) -> Optional[str]:
    """
    Infer the recipe from the model_path.

    :param model_path: The path to the model to load. It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model ID
    :return: The path to the recipe file if found, None otherwise.
    """
    model_path = model_path.as_posix() if isinstance(model_path, Path) else model_path

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        # Model path is a local path to the model directory or file
        model_path = (
            os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        )
        recipe = os.path.join(model_path, RECIPE_FILE_NAME)

        if os.path.isfile(recipe):
            logger.info(f"Found recipe in the model_path: {recipe}")
            return recipe
        logger.debug(f"No recipe found in the model_path: {model_path}")
        return None

    # If the model path is a Hugging Face model ID
    recipe = recipe_from_huggingface_model_id(hf_stub=model_path)

    if recipe is None:
        logger.debug("Failed to infer the recipe from the model_path")

    return recipe


def recipe_from_huggingface_model_id(
    hf_stub: str, recipe_file_name: str = RECIPE_FILE_NAME
) -> Optional[str]:
    """
    Attempts to download the recipe from the Hugging Face model ID.

    :param hf_stub: Assumed to be the Hugging Face model ID.
    :param recipe_file_name: The name of the recipe file to download.
     Defaults to RECIPE_FILE_NAME.
    :return: A tuple:
        - The path to the recipe file if found, None otherwise.
        - True if hf_stub is a valid Hugging Face model ID, False otherwise.
    """
    model_id_url = os.path.join(HUGGINGFACE_CO_URL_HOME, hf_stub)
    request = requests.head(model_id_url)

    if request.status_code != 200:
        logger.debug(
            (
                "hf_stub is not a valid Hugging Face model ID. ",
                "Skipping recipe resolution.",
            )
        )
        return None

    try:
        recipe = hf_hub_download(repo_id=hf_stub, filename=recipe_file_name)
        logger.info(f"Found recipe: {recipe_file_name} for model ID: {hf_stub}.")
    except Exception as e:  # TODO: narrow acceptable exceptions
        logger.debug(
            (
                f"Unable to find recipe {recipe_file_name} "
                f"for model ID: {hf_stub}: {e}."
                "Skipping recipe resolution."
            )
        )
        recipe = None

    return recipe


def resolve_processor_from_model_args(model_args: "ModelArguments"):
    # silently assign tokenizer to processor
    if model_args.tokenizer:
        if model_args.processor:
            raise ValueError("Cannot use both a tokenizer and processor")
        model_args.processor = model_args.tokenizer
    model_args.tokenizer = None
