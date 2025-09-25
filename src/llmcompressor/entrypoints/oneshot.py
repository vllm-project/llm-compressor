"""
Oneshot compression entrypoint for post-training model optimization.

Provides the main oneshot compression entry point for applying
quantization, pruning, and other compression techniques to pre-trained
models without additional training. Supports calibration-based compression
with various pipeline configurations for efficient model optimization.
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from loguru import logger
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from llmcompressor.args import parse_args
from llmcompressor.core.session_functions import active_session
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.entrypoints.utils import post_process, pre_process
from llmcompressor.pipelines import CalibrationPipeline

__all__ = ["Oneshot", "oneshot"]

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


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
        log_dir: Optional[str] = "sparse_logs",
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
        :param log_dir: Path to save logs during oneshot run.
            Nothing is logged to file if None.
        """
        # Set up logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.add(
                f"{log_dir}/oneshot_{date_str}.log",
                level="DEBUG",
            )

        model_args, dataset_args, recipe_args, _, output_dir = parse_args(**kwargs)

        self.model_args = model_args
        self.dataset_args = dataset_args
        self.recipe_args = recipe_args
        self.output_dir = output_dir

        # initialize the model and processor
        pre_process(model_args, dataset_args, output_dir)

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
        session.reset()

        # (Helen INFERENG-661): validate recipe modifiers before intialization
        session.initialize(
            model=self.model,
            start=-1,
            recipe=self.recipe,
            recipe_stage=recipe_stage,
            recipe_args=self.recipe_args.recipe_args,
            calib_data=calibration_dataloader,
        )
        user_pipeline = self.dataset_args.pipeline
        modifiers = session.lifecycle.recipe.modifiers
        pipeline = CalibrationPipeline.from_modifiers(modifiers, user=user_pipeline)
        pipeline(
            self.model,
            calibration_dataloader,
            self.dataset_args,
        )

        session.finalize()


def oneshot(
    # Model arguments
    model: Union[str, PreTrainedModel],
    distill_teacher: Optional[str] = None,
    config_name: Optional[str] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    precision: str = "auto",
    tie_word_embeddings: bool = False,
    trust_remote_code_model: bool = False,
    save_compressed: bool = True,
    model_revision: str = "main",
    # Recipe arguments
    recipe: Optional[Union[str, List[str]]] = None,
    recipe_args: Optional[List[str]] = None,
    clear_sparse_session: bool = False,
    stage: Optional[str] = None,
    # Dataset arguments
    dataset: Optional[Union[str, "Dataset", "DatasetDict"]] = None,
    dataset_config_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    splits: Optional[Union[str, List, Dict]] = None,
    num_calibration_samples: int = 512,
    shuffle_calibration_samples: bool = True,
    max_seq_length: int = 384,
    pad_to_max_length: bool = True,
    text_column: str = "text",
    concatenate_data: bool = False,
    streaming: bool = False,
    overwrite_cache: bool = False,
    preprocessing_num_workers: Optional[int] = None,
    min_tokens_per_module: Optional[float] = None,
    calibrate_moe_context: bool = False,
    quantization_aware_calibration: bool = True,
    # Miscellaneous arguments
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = "sparse_logs",
    **kwargs,
) -> PreTrainedModel:
    """
    Performs oneshot calibration on a model.

    # Model arguments
    :param model: A pretrained model identifier from huggingface.co/models or a path
        to a local model. Required parameter.
    :param distill_teacher: Teacher model (a trained text generation model)
        for distillation.
    :param config_name: Pretrained config name or path if not the same as
        model_name.
    :param tokenizer: Pretrained tokenizer name or path if not the same as
        model_name.
    :param processor: Pretrained processor name or path if not the same as
        model_name.
    :param cache_dir: Where to store the pretrained data from
        huggingface.co.
    :param use_auth_token: Whether to use Hugging Face auth token for private
        models.
    :param precision: Precision to cast model weights to, default to auto.
    :param tie_word_embeddings: Whether the model's input and output word embeddings
        should be tied.
    :param trust_remote_code_model: Whether to allow for custom models to execute
        their own modeling files.
    :param save_compressed: Whether to compress sparse models during save.
    :param model_revision: The specific model version to use (can be branch name,
        tag, or commit id).

    # Recipe arguments
    :param recipe: Path to a LLM Compressor sparsification recipe.
    :param recipe_args: List of recipe arguments to evaluate, in the
        format "key1=value1", "key2=value2".
    :param clear_sparse_session: Whether to clear CompressionSession/
        CompressionLifecycle data between runs.
    :param stage: The stage of the recipe to use for oneshot.

    # Dataset arguments
    :param dataset: The name of the dataset to use (via the datasets
        library).
    :param dataset_config_name: The configuration name of the dataset
        to use.
    :param dataset_path: Path to a custom dataset. Supports json, csv, dvc.
    :param splits: Optional percentages of each split to download.
    :param num_calibration_samples: Number of samples to use for one-shot
        calibration.
    :param shuffle_calibration_samples: Whether to shuffle the dataset before
        calibration.
    :param max_seq_length: Maximum total input sequence length after tokenization.
    :param pad_to_max_length: Whether to pad all samples to `max_seq_length`.
    :param text_column: Key to use as the `text` input to tokenizer/processor.
    :param concatenate_data: Whether to concatenate datapoints to fill
        max_seq_length.
    :param streaming: True to stream data from a cloud dataset.
    :param overwrite_cache: Whether to overwrite the cached preprocessed datasets.
    :param preprocessing_num_workers: Number of processes for
        preprocessing.
    :param min_tokens_per_module: Minimum percentage of tokens per
        module, relevant for MoE models.
    :param calibrate_moe_context: If during calibration, the MoE context should be
        enabled for the given model. This usually involves updating all MoE modules
        in the model for the duration of calibration.
    :param quantization_aware_calibration: Whether to enable quantization-aware
        calibration in the sequential pipeline. When True, quantization is applied
        during forward pass in calibration. When False, quantization is disabled
        during forward pass in calibration. Default is set to True.

    # Miscellaneous arguments
    :param output_dir: Path to save the output model after calibration.
        Nothing is saved if None.
    :param log_dir: Path to save logs during oneshot run.
        Nothing is logged to file if None.

    :return: The calibrated PreTrainedModel
    """

    # pass all args directly into Oneshot
    local_args = {
        k: v for k, v in locals().items() if k not in ("local_args", "kwargs")
    }
    one_shot = Oneshot(**local_args, **kwargs)
    one_shot()

    return one_shot.model
