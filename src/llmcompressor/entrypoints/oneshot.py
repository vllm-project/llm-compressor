from typing import Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from llmcompressor.args import parse_args
from llmcompressor.core.session_functions import active_session
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.entrypoints.utils import post_process, pre_process

__all__ = ["Oneshot", "oneshot"]


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
        - `dataset_args`: Arguments for dataset-related configurations, such as
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

        __call__(**kwargs):
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

    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the `Oneshot` class with provided arguments.

        Parses the input keyword arguments into `model_args`, `dataset_args`, and
        `recipe_args`. Performs preprocessing to initialize the model and
        tokenizer/processor.

        :param model_args: ModelArguments parameters, responsible for controlling
            model loading and saving logic
        :param dataset_args: DatasetArguments parameters, responsible for controlling
            dataset loading, preprocessing and dataloader loading
        :param recipe_args: RecipeArguments parameters, responsible for containing
            recipe-related parameters
        :param output_dir: Path to save the output model after carrying out oneshot

        """
        model_args, dataset_args, recipe_args, _, output_dir = parse_args(**kwargs)

        self.model_args = model_args
        self.dataset_args = dataset_args
        self.recipe_args = recipe_args
        self.output_dir = output_dir

        # initialize the model and processor
        pre_process(model_args)

        # Set instance attributes
        self.model = self.model_args.model
        self.processor = self.model_args.processor
        self.recipe = self.recipe_args.recipe

    def __call__(self):
        """
        Performs one-shot calibration.

        This method prepares a calibration dataloader using dataset arguments and
        applies recipe-based modifiers to optimize the model. The lifecycle actions
        are executed sequentially, and the modified model is saved during
        postprocessing.

        """

        calibration_dataloader = get_calibration_dataloader(
            self.dataset_args, self.processor
        )
        self.apply_recipe_modifiers(
            calibration_dataloader=calibration_dataloader,
            recipe_stage=self.recipe_args.stage,
        )
        post_process(
            model_args=self.model_args,
            recipe_args=self.recipe_args,
            output_dir=self.output_dir,
        )

    def apply_recipe_modifiers(
        self,
        calibration_dataloader: Optional[DataLoader],
        recipe_stage: Optional[str] = None,
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
            recipe_stage=recipe_stage,
        )

        session.reset()
        session.initialize(**session_kwargs)
        session.finalize(**session_kwargs)


def oneshot(**kwargs) -> PreTrainedModel:
    one_shot = Oneshot(**kwargs)
    one_shot()

    return one_shot.model
