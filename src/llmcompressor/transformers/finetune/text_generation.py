#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/transformers
# vllm-project: no copyright


from pathlib import PosixPath

from compressed_tensors.utils.helpers import deprecated
from loguru import logger

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.core import reset_session
from llmcompressor.pytorch.model_load.helpers import save_checkpoint
from llmcompressor.recipe import Recipe, StageRunType
from llmcompressor.transformers.finetune.runner import StageRunner
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.utils.fsdp.helpers import is_fsdp_model


@deprecated(
    message=(
        "`from llmcompressor.transformers import oneshot` is deprecated, "
        "please use `from llmcompressor import oneshot`."
    )
)
def oneshot(**kwargs) -> None:
    from llmcompressor import oneshot

    oneshot(**kwargs)


@deprecated(
    message=(
        "`from llmcompressor import train` is deprecated, "
        "please use `from llmcompressor import train`."
    )
)
def train(**kwargs):
    from llmcompressor import train

    train(**kwargs)


def apply(**kwargs):
    """
    CLI entrypoint for any of training, oneshot
    """
    from llmcompressor.args import parse_args

    model_args, dataset_args, recipe_args, training_args, _ = parse_args(
        include_training_args=True, **kwargs
    )

    training_args.run_stages = True
    report_to = kwargs.get("report_to", None)
    if report_to is None:  # user didn't specify any reporters
        # get rid of the reporters inferred from hugging face
        training_args.report_to = []
    main(model_args, dataset_args, recipe_args, training_args)


def compress(**kwargs):
    apply(**kwargs)


def main(
    model_args: ModelArguments,
    dataset_args: DatasetArguments,
    recipe_args: RecipeArguments,
    training_args: TrainingArguments,
):
    """
    Main entrypoint for finetuning text generation models. A model can be loaded from
    Hugging Face or disk, and resuming training from a checkpoint is supported.

    Lifecycle:
        - SparseAutoModel.text_generation_from_pretrained if model provided as
            string for model and teacher
        - AutoTokenizer.from_pretrained() if tokenizer provided as
            string for tokenizer
        - StageRunner.populate_datasets()
        - Trainer()
            - SessionMixIn()
            - HFTransformersTrainer()
        - StageRunner.train() and/or  oneshot()


    :param model_args: Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from
    :param dataset_args: Arguments pertaining to what data we are
        going to input our model for training
    :param training_args: Arguments pertaining to training loop configuration
    """
    from llmcompressor.args import TrainingArguments
    from llmcompressor.entrypoints.utils import (
        initialize_model_from_path,
        initialize_processor_from_path,
    )

    # Temporary warning, to be removed
    if model_args.tie_word_embeddings is True:
        logger.warning(
            "The tie_word_embeddings flag is by default set to False. "
            "This guarantees that the one-shot algorithm saves the final "
            "weights without errors. Detected tie_word_embeddings=True. "
            "This may cause issues with the one-shot algorithm on save. "
        )

    # Setup based on stage types if running stage mode
    if training_args.run_stages and recipe_args.recipe is not None:
        recipe_obj = Recipe.create_instance(recipe_args.recipe)
        for stage in recipe_obj.stages:
            run_type = stage.infer_run_type()
            if run_type is StageRunType.ONESHOT:
                training_args.do_oneshot = True
            elif run_type is StageRunType.TRAIN:
                training_args.do_train = True

    # Summary on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    teacher = model_args.distill_teacher
    # distill TODO: support for different processor for teacher?

    model = model_args.model
    if isinstance(model, str) or isinstance(model, PosixPath):
        model, teacher = initialize_model_from_path(
            model_args,
            training_args,
        )
    # patch a shared tensor bug in HF transformers
    # https://github.com/huggingface/transformers/issues/33689
    patch_tied_tensors_bug(model)

    if teacher is not None:
        teacher.eval()

    processor = model_args.processor
    if isinstance(processor, str) or processor is None:
        processor = initialize_processor_from_path(model_args, model, teacher)

    # Load datasets
    stage_runner = StageRunner(
        model_args=model_args,
        dataset_args=dataset_args,
        training_args=training_args,
        recipe_args=recipe_args,
    )
    add_labels = training_args.do_train or training_args.run_stages
    stage_runner.populate_datasets(processor=processor, add_labels=add_labels)
    train_dataset = stage_runner.get_dataset_split("train")
    calib_dataset = stage_runner.get_dataset_split("calibration")

    trainer = Trainer(
        model_init=lambda: model,
        teacher=teacher,
        recipe=recipe_args.recipe,
        recipe_args=recipe_args.recipe_args,
        args=training_args,
        model_args=model_args,
        dataset_args=dataset_args,
        train_dataset=train_dataset or calib_dataset,
        processing_class=processor,
        data_collator=dataset_args.data_collator,
    )

    # wrap model.save_pretrained
    if is_fsdp_model(model):
        raise NotImplementedError(
            "FSDP models are not supported in the current release but will be "
            "suported in future releases of LLM Compressor"
        )
    else:
        modify_save_pretrained(model)

    stage_runner.trainer = trainer

    # alternating Training/One-shot
    if training_args.run_stages:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        stage_runner.run_sequential_stages(model, checkpoint)

        # exit immediately
        return

    # save if model was provided as a string or custom output_dir was set
    if isinstance(model_args.model, str) or (
        training_args.output_dir
        != TrainingArguments.__dataclass_fields__["output_dir"].default
        and trainer.accelerator.is_main_process
    ):
        save_checkpoint(
            save_path=training_args.output_dir,
            model=model,
            processor=processor,
            save_safetensors=True,
            save_compressed=model_args.save_compressed,
        )
    trainer.accelerator.wait_for_everyone()

    # Clean up the CompressionSession before exit if requested
    if recipe_args.clear_sparse_session:
        reset_session()


if __name__ == "__main__":
    apply()
