"""
Training entrypoint for fine-tuning models with compression support.

Provides the main training entry point that supports both vanilla
fine-tuning and compression-aware training workflows. Integrates with
HuggingFace transformers and supports knowledge distillation, pruning,
and quantization during the training process.
"""

import math
import os

from compressed_tensors.utils import deprecated
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.args import parse_args
from llmcompressor.core.session_functions import active_session
from llmcompressor.datasets.utils import get_processed_dataset
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.utils.dev import dispatch_for_generation

from .utils import post_process, pre_process


@deprecated(
    message=(
        "Training support will be removed in future releases. Please use "
        "the llmcompressor Axolotl integration for fine-tuning "
        "https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open"  # noqa: E501
    )
)
def train(**kwargs) -> PreTrainedModel:
    """
    Fine-tuning entrypoint that supports vanilla fine-tuning and
    knowledge distillation for compressed model using `oneshot`.


    This entrypoint is responsible the entire fine-tuning lifecycle, including
    preprocessing (model and tokenizer/processor initialization), fine-tuning,
    and postprocessing (saving outputs). The intructions for fine-tuning compressed
    model can be specified by using a recipe.

    - **Input Keyword Arguments:**
        `kwargs` are parsed into:
        - `model_args`: Arguments for loading and configuring a pretrained model
          (e.g., `AutoModelForCausalLM`).
        - `dataset_args`: Arguments for dataset-related configurations, such as
          calibration dataloaders.
        - `recipe_args`: Arguments for defining and configuring recipes that specify
          optimization actions.
        - `training_args`: rguments for defining and configuring training parameters

        Parsers are defined in `src/llmcompressor/args/`.

    - **Lifecycle Overview:**
        The fine-tuning lifecycle consists of three steps:
        1. **Preprocessing**:
            - Instantiates a pretrained model and tokenizer/processor.
            - Ensures input and output embedding layers are untied if they share
              tensors.
            - Patches the model to include additional functionality for saving with
              quantization configurations.
        2. **Training**:
            - Finetunes the model using a global `CompressionSession` and applies
              recipe-defined modifiers (e.g., `ConstantPruningModifier`,
                `OutputDistillationModifier`)
        3. **Postprocessing**:
            - Saves the model, tokenizer/processor, and configuration to the specified
              `output_dir`.

    - **Usage:**
        ```python
        train(model=model, recipe=recipe, dataset=dataset)

        ```

    """
    model_args, dataset_args, recipe_args, training_args, output_dir = parse_args(
        include_training_args=True, **kwargs
    )

    pre_process(model_args, dataset_args, output_dir)
    dispatch_for_generation(model_args.model)  # train is dispatched same as generation

    processed_dataset = get_processed_dataset(
        dataset_args=dataset_args,
        processor=model_args.processor,
    )
    training_dataset = processed_dataset.get("train")

    # create output dir for stages
    original_output_dir = output_dir = training_args.output_dir
    if all([output_dir, recipe_args, getattr(recipe_args, "stage", None)]):
        output_dir = os.path.join(original_output_dir, recipe_args.stage)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # update output dir in training args
        logger.info(
            f"Stage detected for training. Updating output dir to: {output_dir}"
        )
        training_args.output_dir = output_dir

    trainer = Trainer(
        model=model_args.model,
        teacher=model_args.distill_teacher,
        recipe=recipe_args.recipe,
        recipe_args=recipe_args.recipe_args,
        args=training_args,
        model_args=model_args,
        dataset_args=dataset_args,
        train_dataset=training_dataset,
        processing_class=model_args.processor,
        data_collator=dataset_args.data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    logger.info("*** Train ***")

    session = active_session()
    session.reset()
    train_result = trainer.train(
        resume_from_checkpoint=checkpoint,
        stage=recipe_args.stage,
    )

    # return output
    metrics = train_result.metrics
    metrics["train_samples"] = len(training_dataset)
    metrics["perplexity"] = math.exp(metrics["train_loss"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # this includes saving the state, optimizer and scheduler
    # TODO: support all save args, not just skip_sparsity_compression_stats
    trainer.save_model(
        output_dir=training_args.output_dir, skip_sparsity_compression_stats=False
    )

    post_process(recipe_args=recipe_args)
    training_args.output_dir = original_output_dir

    return model_args.model
