"""
Helper variables and functions for integrating LLM Compressor with
huggingface/transformers flows
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import requests
from huggingface_hub import HUGGINGFACE_CO_URL_HOME, hf_hub_download
from loguru import logger
from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint

if TYPE_CHECKING:
    from llmcompressor.transformers import ModelArguments, TrainingArguments

__all__ = [
    "RECIPE_FILE_NAME",
    "detect_last_checkpoint",
    "is_model_quantized_from_path",
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


def is_model_quantized_from_path(path: str):
    """
    Determine if model is quantized based on the config
    """
    config = AutoConfig.from_pretrained(path)
    if config is not None:
        if hasattr(config, "quantization_config"):
            return True
        return False


def resolve_recipe(
    model_path: Union[str, Path],
    recipe: Union[str, Path, None] = None,
) -> Union[str, None]:
    """
    Resolve the recipe to apply to the model.
    :param recipe: the recipe to apply to the model.
        It can be one of the following:
        - None
            This means that we are not either not applying
            any recipe and allowing the model to potentially
            infer the appropriate pre-existing recipe
            from the model_path
        - a path to the recipe file
            This can be a string or Path object pointing
            to a recipe file. If the specified recipe file
            is different from the potential pre-existing
            recipe for that model (stored in the model_path),
            the function will raise an warning
        - name of the recipe file (e.g. "recipe.yaml")
            Recipe file name specific is assumed to be stored
            in the model_path
        - a string containing the recipe
            Needs to adhere to the SparseML recipe format

    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id

    :return: the resolved recipe
    """

    if recipe is None:
        return infer_recipe_from_model_path(model_path)

    elif os.path.isfile(recipe):
        # recipe is a path to a recipe file
        return resolve_recipe_file(recipe, model_path)

    elif os.path.isfile(os.path.join(model_path, recipe)):
        # recipe is a name of a recipe file
        recipe = os.path.join(model_path, recipe)
        return resolve_recipe_file(recipe, model_path)

    elif isinstance(recipe, str):
        # recipe is a string containing the recipe
        logger.debug(
            "Applying the recipe string directly to the model, without "
            "checking for a potential existing recipe in the model_path."
        )
        return recipe

    logger.info(
        "No recipe requested and no default recipe "
        f"found in {model_path}. Skipping recipe resolution."
    )
    return None


def infer_recipe_from_model_path(model_path: Union[str, Path]) -> Optional[str]:
    """
    Infer the recipe from the model_path.
    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id
    :return the path to the recipe file if found, None otherwise
    """
    model_path = model_path.as_posix() if isinstance(model_path, Path) else model_path

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        # model_path is a local path to the model directory or model file
        # attempting to find the recipe in the model_directory
        model_path = (
            os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        )
        recipe = os.path.join(model_path, RECIPE_FILE_NAME)
        if os.path.isfile(recipe):
            logger.info(f"Found recipe in the model_path: {recipe}")
            return recipe
        logger.debug(f"No recipe found in the model_path: {model_path}")
        return None

    recipe = recipe_from_huggingface_model_id(model_path)[0]

    if recipe is None:
        logger.info("Failed to infer the recipe from the model_path")
    return recipe


def recipe_from_huggingface_model_id(
    model_path: str, RECIPE_FILE_NAME: str = RECIPE_FILE_NAME
) -> Tuple[Optional[str], bool]:
    """
    Attempts to download the recipe from the huggingface model id.

    :param model_path: Assumed to be the huggingface model id.
        If it is not, this function will return None.
    :param RECIPE_FILE_NAME: The name of the recipe file to download.
        Defaults to RECIPE_FILE_NAME.
    :return: tuple:
        - the path to the recipe file if found, None otherwise
        - True if model_path is a valid huggingface model id, False otherwise
    """
    model_id = os.path.join(HUGGINGFACE_CO_URL_HOME, model_path)
    request = requests.get(model_id)
    if not request.status_code == 200:
        logger.debug(
            "model_path is not a valid huggingface model id. "
            "Skipping recipe resolution."
        )
        return None, False

    logger.info(
        "model_path is a huggingface model id. "
        "Attempting to download recipe from "
        f"{HUGGINGFACE_CO_URL_HOME}"
    )
    try:
        recipe = hf_hub_download(repo_id=model_path, filename=RECIPE_FILE_NAME)
        logger.info(f"Found recipe: {RECIPE_FILE_NAME} for model id: {model_path}.")
    except Exception as e:
        logger.info(
            f"Unable to to find recipe {RECIPE_FILE_NAME} "
            f"for model id: {model_path}: {e}. "
            "Skipping recipe resolution."
        )
        recipe = None
    return recipe, True


def resolve_recipe_file(
    requested_recipe: Union[str, Path], model_path: Union[str, Path]
) -> Union[str, Path, None]:
    """
    Given the requested recipe and the model_path, return the path to the recipe file.

    :param requested_recipe. Is a full path to the recipe file
    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id
    :return the path to the recipe file if found, None otherwise
    """
    # preprocess arguments so that they are all strings
    requested_recipe = (
        requested_recipe.as_posix()
        if isinstance(requested_recipe, Path)
        else requested_recipe
    )
    model_path = model_path.as_posix() if isinstance(model_path, Path) else model_path
    model_path = (
        os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    )

    if not os.path.isdir(model_path):
        default_recipe, model_exists = recipe_from_huggingface_model_id(model_path)
        if not model_exists:
            raise ValueError(f"Unrecognized model_path: {model_path}")

        if not default_recipe == requested_recipe and default_recipe is not None:
            logger.warning(
                f"Attempting to apply recipe: {requested_recipe} "
                f"to the model at: {model_path}, "
                f"but the model already has a recipe: {default_recipe}. "
                f"Using {requested_recipe} instead."
            )
        return requested_recipe

    # pathway for model_path that is a directory
    default_recipe = os.path.join(model_path, RECIPE_FILE_NAME)
    default_recipe_exists = os.path.isfile(default_recipe)
    default_and_request_recipes_identical = os.path.samefile(
        default_recipe, requested_recipe
    )

    if (
        default_recipe_exists
        and requested_recipe
        and not default_and_request_recipes_identical
    ):
        logger.warning(
            f"Attempting to apply recipe: {requested_recipe} "
            f"to the model located in {model_path}, "
            f"but the model already has a recipe stored as {default_recipe}. "
            f"Using {requested_recipe} instead."
        )

    elif not default_recipe_exists and requested_recipe:
        logger.warning(
            f"Attempting to apply {requested_recipe} "
            f"to the model located in {model_path}."
            "However, it is expected that the model "
            f"has its target recipe stored as {default_recipe}."
            "Applying any recipe before the target recipe may "
            "result in unexpected behavior."
            f"Applying {requested_recipe} nevertheless."
        )

    elif default_recipe_exists:
        logger.info(f"Using the default recipe: {requested_recipe}")

    return requested_recipe
