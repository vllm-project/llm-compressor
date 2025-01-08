from pathlib import PosixPath
from typing import Optional

from loguru import logger
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_calibration_dataloader,
)
from llmcompressor.transformers.finetune.model_args import OneshotModelArguments
from llmcompressor.transformers.finetune.text_generation import (
    initialize_oneshot_model,
    initialize_processor_from_path,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.utils.recipe_args import RecipeArguments


class Oneshot:
    """
    Class responsisble for carrying out oneshot calibration

    Lifecycle:
        - Instantiate CompressionLifecycle that is responsible for applying the recipe
        - Carry out pre-processing - model, tokenizer/processor instantiation,
        untie shared tensors, wrap model.save_pretrained to save models in
        compressed_tensors format for vllm inference
        - Get calibration dataloader for dataset to calibrate the scales and zero points
        - Applying recipe modifiers using the calibration dataloader
        - Carry out post-processing - save the model in compressed_tensors format
        if the model was provided as a string or custom output_dir was set

    Usage:

    ```python
    oneshot_calibrator = Oneshot(model=model, recipe=recipe, dataset=dateset)
    oneshot_calibrator.run()

    model = oneshot_calibrator.model
    tokenizer_or_processor = oneshot_calibrator.tokenizer_or_processor
    recipe = oneshot_calibrator.recipe

    ```
    """

    def __init__(self, **kwargs):
        self.model_args, self.data_args, self.recipe_args = parse_oneshot_args(**kwargs)

        # Singleton for consecutive oneshot calls to keep applied recipe history
        self.lifecycle = CompressionLifecycle()

        # model, tokenizer/processor instantiation
        self._pre_process()

        self.model = self.model_args.model
        self.tokenizer_or_processor = self.model_args.processor
        self.recipe = self.recipe_args.recipe
        self.modifiers = self.lifecycle.modifiers

    def run(self):
        """Carry out oneshot calibration"""
        calibration_dataloader = get_calibration_dataloader(
            self.data_args, self.tokenizer_or_processor
        )

        self.apply_recipe_modifiers(calibration_dataloader=calibration_dataloader)

        self._post_process()

    def apply_recipe_modifiers(self, calibration_dataloader: Optional[DataLoader]):
        """Apply recipe modifiers to the model"""
        self.lifecycle.initialize(
            model=self.model,
            recipe=self.recipe,
            recipe_args=self.recipe_args.recipe_args,
            calib_data=calibration_dataloader,
            start=-1,  # oneshot specific arg
            copy_data=False,
            min_tokens_per_module=self.min_tokens_per_module,
        )

        self.lifecycle.finalize(
            model=self.model,
            recipe=self.recipe,
            recipe_args=self.recipe_args.recipe_args,
            calib_data=calibration_dataloader,
            start=-1,  # oneshot specific arg
            copy_data=False,
            min_tokens_per_module=self.min_tokens_per_module,
        )

    def _pre_process(self):
        """Preprocess model and tokenizer/processor"""
        if self.model_args.tie_word_embeddings is True:
            logger.debug(
                "The tie_word_embeddings flag is by default set to False. "
                "This guarantees that the one-shot algorithm saves the final "
                "weights without errors. Detected tie_word_embeddings=True. "
                "This may cause issues with the one-shot algorithm on save. "
            )

        model = self.model_args.model
        if isinstance(model, str) or isinstance(model, PosixPath):
            model = initialize_oneshot_model(self.model_args)

        # patch a shared tensor bug in HF transformers
        # https://github.com/huggingface/transformers/issues/33689
        patch_tied_tensors_bug(model)

        # on save, convert the model in a compressed_tensors format for vllm inference
        modify_save_pretrained(model)

        self.model_args.model = model

        processor = self.model_args.processor
        if isinstance(processor, str) or processor is None:
            self.model_args.processor = initialize_processor_from_path(
                self.model_args, model
            )

        if self.data_args is not None:
            self.min_tokens_per_module = self.data_args.min_tokens_per_module

    def _post_process(self):
        """Save model if custom path was set and reset lifecycle if requested"""
        # save if model was provided as a string or custom output_dir was set
        if isinstance(self.model_args.model, str) or (
            self.model_args.output_dir
            != OneshotModelArguments.__dataclass_fields__["output_dir"].default
        ):
            self.model_args.model.save_pretrained(
                self.model_args.output_dir,
                save_compressed=self.model_args.save_compressed,
                stage_modifiers=self.lifecycle.modifiers,
            )
            if self.tokenizer_or_processor is not None:
                self.tokenizer_or_processor.save_pretrained(self.model_args.output_dir)

        # Clean up the CompressionSession before exit if requested
        if self.recipe_args.clear_sparse_session:
            self.reset_lifecycle()

    def reset_lifecycle(self):
        """Reset the CompressionLifecycle"""
        self.lifecycle.reset()


def parse_oneshot_args(**kwargs):
    """Parse oneshot arguments into model_args, data_args and recipe_args"""
    parser = HfArgumentParser(
        (OneshotModelArguments, DataTrainingArguments, RecipeArguments)
    )
    if not kwargs:
        model_args, data_args, recipe_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, recipe_args = parser.parse_dict(kwargs)

    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    if model_args.tokenizer:
        if model_args.processor:
            raise ValueError("Cannot use both a tokenizer and processor")

        logger.debug("Overwriting processor with tokenizer")
        model_args.processor = model_args.tokenizer

    return model_args, data_args, recipe_args
