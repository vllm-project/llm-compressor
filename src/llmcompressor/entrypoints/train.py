import math

from loguru import logger

from llmcompressor.args import parse_args
from llmcompressor.datasets.utils import get_processed_dataset
from llmcompressor.transformers.finetune.trainer import Trainer

from .utils import post_process, pre_process


def train(**kwargs):
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
    model_args, dataset_args, recipe_args, training_args, _ = parse_args(
        include_training_args=True, **kwargs
    )

    pre_process(model_args)

    processed_dataset = get_processed_dataset(
        dataset_args=dataset_args,
        processor=model_args.processor,
    )
    training_dataset = processed_dataset.get("train")

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
    train_result = trainer.train(
        resume_from_checkpoint=checkpoint,
    )

    # return output
    metrics = train_result.metrics
    metrics["train_samples"] = len(training_dataset)
    metrics["perplexity"] = math.exp(metrics["train_loss"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # this includes saving the state, optimizer and scheduler
    trainer.save_model(output_dir=training_args.output_dir)

    post_process(model_args=model_args, output_dir=training_args.output_dir)
