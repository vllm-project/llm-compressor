"""
Utility functions for parsing and processing argument classes.

Provides helper functions for parsing command-line arguments and
configuration dictionaries into structured argument dataclasses used in
LLM compression workflows. Handles argument validation, deprecation
warnings, and processor resolution.
"""

from typing import Tuple

from loguru import logger
from transformers import HfArgumentParser

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.transformers.utils.helpers import resolve_processor_from_model_args


def parse_args(
    include_training_args: bool = False, **kwargs
) -> Tuple[ModelArguments, DatasetArguments, RecipeArguments, TrainingArguments, str]:
    """
    Keyword arguments passed in from `oneshot` or `train` will
    separate the arguments into the following:

        * ModelArguments in
            src/llmcompressor/args/model_args.py
        * DatasetArguments in
            src/llmcompressor/args/dataset_args.py
        * RecipeArguments in
            src/llmcompressor/args/recipe_args.py
        * TrainingArguments in
            src/llmcompressor/args/training_args.py

    ModelArguments, DatasetArguments, and RecipeArguments are used for both
    `oneshot` and `train`. TrainingArguments is only used for `train`.

    """

    # pop output_dir, used as an attr in TrainingArguments, where oneshot is not used
    output_dir = kwargs.pop("output_dir", None)

    parser_args = (ModelArguments, DatasetArguments, RecipeArguments)
    if include_training_args:
        parser_args += (TrainingArguments,)

    parser = HfArgumentParser(parser_args)
    parsed_args = parser.parse_dict(kwargs)

    training_args = None
    if include_training_args:
        model_args, dataset_args, recipe_args, training_args = parsed_args
        if output_dir is not None:
            training_args.output_dir = output_dir
    else:
        model_args, dataset_args, recipe_args = parsed_args

    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    # raise depreciation warnings
    if dataset_args.remove_columns is not None:
        logger.warn(
            "`remove_columns` argument is depreciated. When tokenizing datasets, all "
            "columns which are invalid inputs the tokenizer will be removed",
            DeprecationWarning,
        )

    # silently assign tokenizer to processor
    resolve_processor_from_model_args(model_args)

    return model_args, dataset_args, recipe_args, training_args, output_dir
