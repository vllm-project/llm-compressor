from pathlib import PosixPath
from typing import Optional, Tuple

from loguru import logger
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, PreTrainedModel

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
from llmcompressor.transformers.utils.helpers import resolve_processor_from_model_args

__all__ = ["Oneshot", "oneshot", "parse_oneshot_args"]


class Oneshot:
    """
    Class responsible for carrying out one-shot calibration on a pretrained model.

    This class handles the entire lifecycle of one-shot calibration, including
    preprocessing (model and tokenizer/processor initialization), model optimization
    (quantization or sparsification), and postprocessing (saving outputs). The
    intructions for model optimization can be specified by using a recipe.

    - **Input Keyword Arguments:**
        `kwargs` are parsed into:
        - `model_args`: Arguments for loading and configuring a pretrained model
          (e.g., `AutoModelForCausalLM`).
        - `data_args`: Arguments for dataset-related configurations, such as
          calibration dataloaders.
        - `recipe_args`: Arguments for defining and configuring recipes that specify
          optimization actions.

        Parsers are defined in `src/llmcompressor/args/`.

    - **Lifecycle Overview:**
        The oneshot calibration lifecycle consists of three steps:
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
        oneshot()

        # Access the processed components
        model = oneshot.model
        processor = oneshot.processor
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

        apply_recipe_modifiers(calibration_dataloader, **kwargs):
            Applies lifecycle actions (e.g., `initialize`, `finalize`) using modifiers
            defined in the recipe. Each action is executed via the global
            `CompressionSession`.

        _pre_process():
            Handles preprocessing steps, including model initialization,
            tokenizer/processor setup, and resolving tied embedding issues.

        check_tied_embeddings():
            Logs a warning if `tie_word_embeddings=True`, which may interfere with
            saving in the one-shot workflow.

        _post_process():
            Executes postprocessing steps such as saving the model and resetting
            lifecycle actions, especially when a custom `output_dir` is specified.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the `Oneshot` class with provided arguments.

        Parses the input keyword arguments into `model_args`, `data_args`, and
        `recipe_args`. Performs preprocessing to initialize the model and
        tokenizer/processor.

        :param model_args: ModelArguments parameters, responsible for controlling
            model loading and saving logic
        :param data_args: DatasetArguments parameters, responsible for controlling
            dataset loading, preprocessing and dataloader loading
        :param recipe_args: RecipeArguments parameters, responsible for containing
            recipe-related parameters
        :param output_dir: Path to save the output model after carrying out oneshot

        """

        model_args, data_args, recipe_args, output_dir = parse_oneshot_args(**kwargs)

        self.model_args = model_args
        self.data_args = data_args
        self.recipe_args = recipe_args
        self.output_dir = output_dir

        # Set instance attributes
        self.model = self.model_args.model
        self.processor = self.model_args.processor
        self.recipe = self.recipe_args.recipe

    @classmethod
    def from_args(
        cls, model_args, data_args, recipe_args, output_dir, do_preprocess: bool = True
    ):
        """
        Used only for the stage runner to populate the args.
        """
        instance = super().__new__(cls)
        instance.model_args = model_args
        instance.data_args = data_args
        instance.recipe_args = recipe_args
        instance.output_dir = output_dir

        # only run for the first oneshot call
        if do_preprocess:
            instance._pre_process()

        # Set instance attributes
        instance.model = instance.model_args.model
        instance.recipe = instance.recipe_args.recipe
        instance.processor = instance.model_args.processor

        return instance

    def __call__(self):
        """
        Performs one-shot calibration.

        This method prepares a calibration dataloader using dataset arguments and
        applies recipe-based modifiers to optimize the model. The lifecycle actions
        are executed sequentially, and the modified model is saved during
        postprocessing.

        """
        # TODO: move back once stage runner is removed
        # Preprocess the model and tokenizer/processor
        self._pre_process()
        self.model = self.model_args.model
        self.recipe = self.recipe_args.recipe
        self.processor = self.model_args.processor

        calibration_dataloader = get_calibration_dataloader(
            self.data_args, self.processor
        )
        self.apply_recipe_modifiers(
            calibration_dataloader=calibration_dataloader,
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
        if self.processor is not None:
            self.processor.save_pretrained(self.output_dir)

    def apply_recipe_modifiers(
        self,
        calibration_dataloader: Optional[DataLoader],
    ):
        """
        Applies recipe modifiers to the model during the lifecycle.

        The modifiers are defined in the recipe and executed via lifecycle actions
        (`initialize`, `finalize`) through the global `CompressionSession`.


        :param: calibration_dataloader: Dataloader for calibration data.

        Raises:
            RuntimeError: If any modifier fails during execution.
        """

        session = active_session()

        session_kwargs = dict(
            model=self.model,
            recipe=self.recipe,
            recipe_args=self.recipe_args.recipe_args,
            calib_data=calibration_dataloader,
            start=-1,  # oneshot-specific argument
            copy_data=False,
            min_tokens_per_module=getattr(self, "min_tokens_per_module", None),
        )

        session.initialize(**session_kwargs)
        session.finalize(**session_kwargs)

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
        self.check_tied_embeddings()

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
            # TODO: move to init once stage runner is removed
            self.processor = self.model_args.processor

        # Set minimum tokens per module if data arguments are provided
        if self.data_args:
            self.min_tokens_per_module = self.data_args.min_tokens_per_module

    def check_tied_embeddings(self):
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
        if self.output_dir is not None:
            self.save()
            return

        logger.warning(
            "Optimized model not saved. To save, please provide",
            "`output_dir` as input arg.",
            "Ex. `oneshot(..., output_dir=...)`",
        )


def oneshot(**kwargs) -> PreTrainedModel:
    one_shot = Oneshot(**kwargs)
    one_shot()

    return one_shot.model


def parse_oneshot_args(
    **kwargs,
) -> Tuple[ModelArguments, DatasetArguments, RecipeArguments, str]:
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
    resolve_processor_from_model_args(model_args)

    return model_args, data_args, recipe_args, output_dir
