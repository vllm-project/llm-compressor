from typing import Optional

from loguru import logger
from transformers import HfArgumentParser

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.transformers.finetune.text_generation import (
    initialize_model_from_path,
    initialize_processor_from_path,
)
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_fsdp_model_save_pretrained,
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.utils.fsdp.helpers import is_fsdp_model


def preprocess(model_args: "ModelArguments"):
    """
    Prepares the model and tokenizer/processor for calibration.
    - Initializes the model if it's specified as a path or string.
    - Applies patches to fix tied tensor issues and modifies `save_pretrained`
        behavior.
    - Initializes the processor if specified as a path or `None`.
    - Sets the minimum tokens per module if `data_args` are provided.
    Raises:
        FileNotFoundError: If the model or processor path is invalid.
    """
    _warn_tied_embeddings(model_args.tie_word_embeddings)

    # Initialize model
    if isinstance(model_args.model, str):
        model_args.model, model_args.distill_teacher = initialize_model_from_path(
            model_args
        )

    patch_tied_tensors_bug(model_args.model)

    # wrap model.save_pretrained
    modify_save_pretrained(model_args.model)

    # Initialize processor
    if isinstance(model_args.processor, (str, type(None))):
        model_args.processor = initialize_processor_from_path(
            model_args, model_args.model
        )


def post_process(
    model_args: "ModelArguments",
    output_dir: Optional[str] = None,
    trainer: Optional[Trainer] = None,
):
    """
    Saves the model and tokenizer/processor to the output directory.

    If the `output_dir` is not the default directory, the method resets lifecycle
    actions. The model is saved in a compressed format if specified in `model_args`.
    Additionally, the tokenizer or processor, if available, is also saved.

    Raises:
        ValueError: If saving fails due to an invalid `output_dir` or other issues.
    """
    if is_fsdp_model(model_args.model):
        assert trainer is not None
        modify_fsdp_model_save_pretrained(
            trainer=trainer, processor=model_args.processor
        )
    else:
        modify_save_pretrained(model_args.model)

    if output_dir is not None:
        model_args.model.save_pretrained(
            output_dir,
            save_compressed=model_args.save_compressed,
        )
        if model_args.processor:
            model_args.processor.save_pretrained(output_dir)


def parse_args(**kwargs):
    """
    Parses kwargs by grouping into model, data or training arg groups:
        * model_args in
            src/llmcompressor/transformers/utils/arg_parser/model_args.py
        * data_args in
            src/llmcompressor/transformers/utils/arg_parser/data_args.py
        * recipe_args in
            src/llmcompressor/transformers/utils/arg_parser/recipe_args.py
        * training_args in
            src/llmcompressor/transformers/utils/arg_parser/training_args.py

    """
    parser = HfArgumentParser(
        (ModelArguments, DatasetArguments, RecipeArguments, TrainingArguments)
    )

    if not kwargs:
        parsed_args = parser.parse_args_into_dataclasses()
    else:
        parsed_args = parser.parse_dict(kwargs)

    model_args, data_args, recipe_args, training_args = parsed_args
    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    # raise depreciation warnings
    if data_args.remove_columns is not None:
        logger.warn(
            "`remove_columns` argument is depreciated. When tokenizing datasets, all "
            "columns which are invalid inputs the tokenizer will be removed",
            DeprecationWarning,
        )

    # silently assign tokenizer to processor
    if model_args.tokenizer:
        if model_args.processor:
            raise ValueError("Cannot use both a tokenizer and processor")
        model_args.processor = model_args.tokenizer
    model_args.tokenizer = None

    return model_args, data_args, recipe_args, training_args


def _warn_tied_embeddings(tie_word_embeddings: bool = False):
    """
    Logs a warning if the model has tied word embeddings.
    The `tie_word_embeddings` flag may cause issues during saving in the one-shot
    calibration workflow due to shared tensor addresses.
    """
    if tie_word_embeddings:
        logger.debug(
            "The tie_word_embeddings flag is by default set to False. "
            "This guarantees that the one-shot algorithm saves the final "
            "weights without errors. Detected tie_word_embeddings=True. "
            "This may cause issues with the one-shot algorithm on save."
        )
