from typing import Optional

from loguru import logger
from transformers import HfArgumentParser

from llmcompressor.args import DatasetArguments, ModelArguments, RecipeArguments
from llmcompressor.core.session_functions import active_session
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_calibration_dataloader,
)
from llmcompressor.transformers.finetune.text_generation import (
    initialize_model_from_path,
    initialize_processor_from_path,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)


def _warn_tied_embeddings(model_args):
    if model_args.tie_word_embeddings:
        logger.debug(
            "The tie_word_embeddings flag is by default set to False. "
            "This guarantees that the one-shot algorithm saves the final "
            "weights without errors. Detected tie_word_embeddings=True. "
            "This may cause issues with the one-shot algorithm on save."
        )


def _preprocess(model_args) -> None:
    """
    Update model and processor on model_args
    """
    _warn_tied_embeddings(model_args)

    # Initialize model
    if isinstance(model_args.model, str):
        # update once intialize_model_from_path PR is merged
        # model_args.model, _ = initialize_model_from_path(model_args)

        _, _, model_args.model = initialize_model_from_path(model_args)

    patch_tied_tensors_bug(model_args.model)
    modify_save_pretrained(model_args.model)

    # Initialize processor
    if isinstance(model_args.processor, (str, type(None))):
        model_args.processor = initialize_processor_from_path(
            model_args, model_args.model
        )

    return model_args.model, model_args.processor


def _post_process(model_args, output_dir):
    if output_dir is not None:
        model_args.model.save_pretrained(
            output_dir,
            save_compressed=model_args.save_compressed,
        )
        if model_args.tokenizer:
            model_args.tokenizer.save_pretrained(output_dir)


def _parse_post_train_args(**kwargs):
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
    output_dir = kwargs.pop("output_dir", None)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, RecipeArguments))

    if not kwargs:

        def _get_output_dir_from_argv() -> Optional[str]:
            import sys

            output_dir = None
            if "--output_dir" in sys.argv:
                index = sys.argv.index("--output_dir")
                sys.argv.pop(index)
                if index < len(sys.argv):  # Check if value exists afer the flag
                    output_dir = sys.argv.pop(index)

            return output_dir

        output_dir = _get_output_dir_from_argv() or output_dir
        parsed_args = parser.parse_args_into_dataclasses()
    else:
        parsed_args = parser.parse_dict(kwargs)

    model_args, data_args, recipe_args = parsed_args

    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    # raise depreciation warnings
    if data_args.remove_columns is not None:
        logger.waning(
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

    return model_args, data_args, recipe_args, output_dir


def run_post_train(
    model,
    recipe,
    recipe_args,
    dataloader,
    min_tokens_per_module: Optional[str] = None,
):
    session = active_session()
    for action in ("initialize", "finalize"):
        session_action = getattr(session, action)
        session_action(
            model=model,
            recipe=recipe,
            recipe_args=recipe_args,
            calib_data=dataloader,
            start=-1,  # post_train-specific argument
            copy_data=False,
            min_tokens_per_module=min_tokens_per_module,
        )


def post_train(
    **kwargs,
):
    model_args, data_args, recipe_args, output_dir = _parse_post_train_args(**kwargs)

    # update model and processor
    _preprocess(model_args)

    calibration_dataloader = get_calibration_dataloader(data_args, model_args.processor)

    run_post_train(
        model=model_args.model,
        recipe=recipe_args.recipe,
        recipe_args=recipe_args.recipe_args,
        dataloader=calibration_dataloader,
        min_tokens_per_module=data_args.min_tokens_per_module,
    )

    _post_process(model_args=model_args, output_dir=output_dir)
