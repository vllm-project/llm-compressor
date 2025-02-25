import math

from loguru import logger

from llmcompressor.args import parse_args
from llmcompressor.datasets.utils import get_processed_dataset
from llmcompressor.transformers.finetune.trainer import Trainer

from .utils import post_process, preprocess


def train(**kwargs):
    model_args, dataset_args, recipe_args, training_args, _ = parse_args(
        include_training_args=True, **kwargs
    )

    preprocess(model_args)

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
