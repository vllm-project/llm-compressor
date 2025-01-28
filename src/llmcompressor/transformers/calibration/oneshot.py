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
    Class responsible for carrying out one-shot calibration on a pretrained model.

    This class handles the entire lifecycle of one-shot calibration, including
    preprocessing (model and tokenizer/processor initialization), model optimization
    (quantization or sparsification), and postprocessing (saving outputs). The
    intructions for model optimization can be specified by using a recipe (fine-grain
    details) or by using a scheme (ex. W4A16, W8A8, W4A8).

    - **Input Keyword Arguments:**
        `kwargs` are parsed into:
        - `model_args`: Arguments for loading and configuring a pretrained model
          (e.g., `AutoModelForCausalLM`).
        - `data_args`: Arguments for dataset-related configurations, such as
          calibration dataloaders.
        - `recipe_args`: Arguments for defining and configuring recipes that specify
          optimization actions.

        Parsers are defined in `src/llmcompressor/transformers/utils/arg_parser`.

    - **Lifecycle Overview:**
        The calibration lifecycle consists of three steps:
        1. **Preprocessing**:
            - Instantiates a pretrained model and tokenizer/processor.
            - Ensures input and output embedding layers are untied if they share
              tensors.
            - Patches the model to include additional functionality for saving with
              quantization configurations.
        2. **Oneshot Calibration**:
            - Optimizes the model using a global `CompressionSession` and applies
              recipe-defined modifiers (e.g., `GPTQModifier`, `SparseGPTModifier`)
        3. **Postprocessing**:
            - Saves the model, tokenizer/processor, and configuration to the specified
              `output_dir`.

    - **Usage:**
        ```python
        oneshot = Oneshot(model=model, recipe=recipe, dataset=dataset)
        oneshot.run()

        # Access the processed components
        model = oneshot.model
        tokenizer_or_processor = oneshot.tokenizer_or_processor
        recipe = oneshot.recipe
        ```

    Methods:
        __init__(**kwargs):
            Initializes the `Oneshot` object by parsing input arguments, performing
            preprocessing, and setting instance attributes.

        run(**kwargs):
            Performs the one-shot calibration process by preparing a calibration
            dataloader, applying recipe modifiers to the model, and executing
            postprocessing steps.

        save():
            Saves the calibrated model and tokenizer/processor to the specified
            `output_dir`. Supports saving in compressed formats based on model
            arguments.

        _apply_recipe_modifiers(calibration_dataloader, **kwargs):
            Applies lifecycle actions (e.g., `initialize`, `finalize`) using modifiers
            defined in the recipe. Each action is executed via the global
            `CompressionSession`.

        _pre_process():
            Handles preprocessing steps, including model initialization,
            tokenizer/processor setup, and resolving tied embedding issues.

        _warn_tied_embeddings():
            Logs a warning if `tie_word_embeddings=True`, which may interfere with
            saving in the one-shot workflow.

        _post_process():
            Executes postprocessing steps such as saving the model and resetting
            lifecycle actions, especially when a custom `output_dir` is specified.
    """

    MODIFIER_LIFECYCLE_ACTIONS = (
        "initialize",
        "finalize",
    )

    def __init__(self, **kwargs):
        """
        Initializes the `Oneshot` class with provided arguments.

        Parses the input keyword arguments into `model_args`, `data_args`, and
        `recipe_args`. Performs preprocessing to initialize the model and
        tokenizer/processor.

        Args:
            kwargs: Arbitrary keyword arguments for model, data, and recipe
            configurations.
        """
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
        """
        Performs one-shot calibration.

        This method prepares a calibration dataloader using dataset arguments and
        applies recipe-based modifiers to optimize the model. The lifecycle actions
        are executed sequentially, and the modified model is saved during
        postprocessing.

        Args:
            kwargs: Additional keyword arguments for the recipe modifiers.
        """
        calibration_dataloader = get_calibration_dataloader(
            self.data_args, self.tokenizer_or_processor
        )
        self._apply_recipe_modifiers(
            calibration_dataloader=calibration_dataloader, **kwargs
        )
        self._post_process()

    def save(self):
        """
        Saves the model and tokenizer/processor to the output directory.

        The model is saved in a compressed format if specified in `model_args`.
        The tokenizer or processor, if available, is also saved.

        Raises:
            ValueError: If saving fails due to an invalid `output_dir` or other issues.
        """
        self.model.save_pretrained(
            self.output_dir,
            save_compressed=self.model_args.save_compressed,
        )
        if self.tokenizer_or_processor:
            self.tokenizer_or_processor.save_pretrained(self.output_dir)

    def _apply_recipe_modifiers(
        self, calibration_dataloader: Optional[DataLoader], **kwargs
    ):
        """
        Applies recipe modifiers to the model during the lifecycle.

        The modifiers are defined in the recipe and executed via lifecycle actions
        (`initialize`, `finalize`) through the global `CompressionSession`.

        Args:
            calibration_dataloader (Optional[DataLoader]): Dataloader for calibration
            data.
            kwargs: Additional arguments for lifecycle actions.

        Raises:
            RuntimeError: If any modifier fails during execution.
        """
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
        """
        Logs a warning if the model has tied word embeddings.

        The `tie_word_embeddings` flag may cause issues during saving in the one-shot
        calibration workflow due to shared tensor addresses.
        """
        if self.model_args.tie_word_embeddings:
            logger.debug(
                "The tie_word_embeddings flag is by default set to False. "
                "This guarantees that the one-shot algorithm saves the final "
                "weights without errors. Detected tie_word_embeddings=True. "
                "This may cause issues with the one-shot algorithm on save."
            )

    def _post_process(self):
        """
        Executes post-calibration steps.

        This method saves the model and resets lifecycle actions if the `output_dir`
        is not the default directory.

        Raises:
            ValueError: If saving fails due to invalid configurations.
        """
        if (
            isinstance(self.model_args.model, str)
            or self.output_dir != DEFAULT_OUTPUT_DIR
        ):
            self.save()
