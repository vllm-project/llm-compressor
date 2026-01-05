"""
Helper variables and functions for integrating LLM Compressor with
huggingface/transformers flows
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from huggingface_hub import (
    _CACHED_NO_EXIST,
    HfApi,
    hf_hub_download,
    try_to_load_from_cache,
)
from loguru import logger
from transformers import AutoConfig

if TYPE_CHECKING:
    from llmcompressor.args import ModelArguments

__all__ = [
    "RECIPE_FILE_NAME",
    "is_model_ct_quantized_from_path",
]

RECIPE_FILE_NAME = "recipe.yaml"


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


def infer_recipe_from_model_path(model_path: str | Path) -> str | None:
    """
    Infer the recipe from the model_path.

    :param model_path: The path to the model to load. It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model ID
    :return: The path to the recipe file if found, None otherwise.
    """
    model_path = (
        model_path.as_posix() if isinstance(model_path, Path) else model_path.strip()
    )
    if model_path == "":
        logger.debug("got path_or_name=<empty string>unable to find recipe")
        return None

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

    # Try to resolve HF model ID to cached location first
    cached_recipe = try_to_load_from_cache(
        repo_id=model_path,
        filename=RECIPE_FILE_NAME,
    )

    if cached_recipe and cached_recipe is not _CACHED_NO_EXIST:
        # Recipe found in cached model
        logger.info(f"Found recipe in cached model: {cached_recipe}")
        return cached_recipe
    # No recipe in cache - fall through to network check

    # If the model path is a Hugging Face model ID
    recipe = recipe_from_huggingface_model_id(hf_stub=model_path)

    if recipe is None:
        logger.debug("Failed to infer the recipe from the model_path")

    return recipe


def recipe_from_huggingface_model_id(
    hf_stub: str, recipe_file_name: str = RECIPE_FILE_NAME
) -> str | None:
    """
    Attempts to download the recipe from the Hugging Face model ID.

    :param hf_stub: Assumed to be the Hugging Face model ID.
    :param recipe_file_name: The name of the recipe file to download.
     Defaults to RECIPE_FILE_NAME.
    :return: A tuple:
        - The path to the recipe file if found, None otherwise.
        - True if hf_stub is a valid Hugging Face model ID, False otherwise.
    """
    # Check if offline mode is enabled
    if os.getenv("HF_HUB_OFFLINE") == "1":
        logger.debug("HF_HUB_OFFLINE is set, skipping recipe download from HuggingFace")
        return None

    # Use custom HF_ENDPOINT
    hf_api = HfApi()
    model_id_url = f"{hf_api.endpoint.rstrip('/')}/{hf_stub}"
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
