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
from llmcompressor.transformers.utils.arg_parser import DEFAULT_OUTPUT_DIR

__all__ = ["Oneshot"]


class Oneshot:
    """
    Class responsible for carrying out oneshot calibration.

    - Input Keyword Arguments:

        kwargs are parsed into
        - model_args
            - responsible for handling Pretrained model loading args
            ex. AutoModelForCausalLM
        - data_args
            - responsible for handling dataset related arguments
        - recipe_args
            - resposible for handling recipe related arguments

        Parsers are defined in
            src/llmcompressor/transformers/utils/arg_parser

    - Lifecycle

        Broken down into three steps
        - Pre-processing
            - Instantiate pretrainined model and tokenizer/processor
            - Untie input and output embedding layers share the same underlying tensor
              which needs to be in a separate address for calibration
            - Wrap the model.save_pretrained model to add
              compressed-tensors quantization config

        - Carrying out oneshot calibration logic
            - Use the global CompressionSession to carry out optimizations
              to the given model.
              Optimizations are based on recipes or preset schemes (ex. W4A16).
              Every optimization method is encapsulated as a Modifier,
               refer to src/llmcompressor/modifiers,
               allowing the session to apply each modifier one by one to the model.

              Ex. Apply just GPTQ -> "GPTQModifier".
               Refer to examples/quantization_w4a16/llama3_example.py
              Ex. Apply sparsification using "SparseGPTModifier" and then
               apply quantization using "GPTQModifier".
               Refer to examples/quantization_2of4_sparse_w4a16/llama7b_sparse_w4a16.py

        - Post-processing
            - Save the model, tokenizer, config and recipe if custom output_dir is
              is specified (not ./output)


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
        **kwargs,
    ):
        self.model_args, self.data_args, self.recipe_args, _, self.output_dir = (
            parse_args(**kwargs)
        )

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
