from pathlib import PosixPath
from typing import Optional

from loguru import logger
from torch.utils.data import DataLoader

from llmcompressor.core.session_functions import active_session
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_calibration_dataloader,
)
from llmcompressor.transformers.finetune.text_generation import (
    initialize_model_from_path,
    initialize_processor_from_path,
    parse_args,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.utils.arg_parser.training_arguments import (
    DEFAULT_OUTPUT_DIR,
)

__all__ = ["Oneshot"]


class Oneshot:
    """
    Class responsible for carrying out oneshot calibration.

    Usage:

    ```python
    oneshot = Oneshot(model=model, recipe=recipe, dataset=dataset)
    oneshot.run()

    model = oneshot.model
    tokenizer_or_processor = oneshot.tokenizer_or_processor
    recipe = oneshot.recipe

    ```
    """

    MODIFIER_LIFECYCLE_ACTIONS = (
        "initialize",
        "finalize",
    )

    def __init__(
        self,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        self.model_args, self.data_args, self.recipe_args, _, output_dir_parser = (
            parse_args(**kwargs)
        )

        self.output_dir = output_dir or output_dir_parser

        # Preprocess the model and tokenizer/processor
        self._pre_process()

        # Set instance attributes
        self.model = self.model_args.model
        self.tokenizer_or_processor = self.model_args.processor
        self.recipe = self.recipe_args.recipe

    def run(self, **kwargs):
        """Perform oneshot calibration"""
        calibration_dataloader = get_calibration_dataloader(
            self.data_args, self.tokenizer_or_processor
        )
        self._apply_recipe_modifiers(
            calibration_dataloader=calibration_dataloader, **kwargs
        )
        self._post_process()

    def save(self):
        """Save the model and tokenizer/processor to the output directory"""
        self.model.save_pretrained(
            self.output_dir,
            save_compressed=self.model_args.save_compressed,
        )
        if self.tokenizer_or_processor:
            self.tokenizer_or_processor.save_pretrained(self.output_dir)

    def _apply_recipe_modifiers(
        self, calibration_dataloader: Optional[DataLoader], **kwargs
    ):
        """Apply recipe modifiers to the model"""
        for action in self.MODIFIER_LIFECYCLE_ACTIONS:
            session = active_session()

            session_action = getattr(session, action)
            session_action(
                model=self.model,
                recipe=self.recipe,
                recipe_args=self.recipe_args.recipe_args,
                calib_data=calibration_dataloader,
                start=-1,  # oneshot-specific argument
                copy_data=False,
                min_tokens_per_module=getattr(self, "min_tokens_per_module", None),
                **kwargs,
            )

    def _pre_process(self):
        """Preprocess model and tokenizer/processor"""
        self._warn_tied_embeddings()

        # Initialize model
        if isinstance(self.model_args.model, (str, PosixPath)):
            self.model_args.model, _ = initialize_model_from_path(self.model_args)

        patch_tied_tensors_bug(self.model_args.model)
        modify_save_pretrained(self.model_args.model)

        # Initialize processor
        if isinstance(self.model_args.processor, (str, type(None))):
            self.model_args.processor = initialize_processor_from_path(
                self.model_args, self.model_args.model
            )

        # Set minimum tokens per module if data arguments are provided
        if self.data_args:
            self.min_tokens_per_module = self.data_args.min_tokens_per_module

    def _warn_tied_embeddings(self):
        if self.model_args.tie_word_embeddings:
            logger.debug(
                "The tie_word_embeddings flag is by default set to False. "
                "This guarantees that the one-shot algorithm saves the final "
                "weights without errors. Detected tie_word_embeddings=True. "
                "This may cause issues with the one-shot algorithm on save"
            )

    def _post_process(self):
        """Save model and reset the lifecycle if requested"""
        if (
            isinstance(self.model_args.model, str)
            or self.output_dir != DEFAULT_OUTPUT_DIR
        ):
            self.save()
