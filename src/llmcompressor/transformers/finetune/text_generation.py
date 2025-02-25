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

import os
import warnings
from pathlib import PosixPath
from typing import Optional

from compressed_tensors.utils.helpers import deprecated
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    PreTrainedModel,
    set_seed,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.core import pre_initialize_structure, reset_session
from llmcompressor.pytorch.model_load.helpers import (
    fallback_to_cpu,
    get_session_model,
    initialize_recipe,
    parse_dtype,
)
from llmcompressor.recipe import Recipe, StageRunType
from llmcompressor.transformers.finetune.runner import StageRunner
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_fsdp_model_save_pretrained,
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.sparsification.sparse_model import (
    get_processor_name_from_model,
)
from llmcompressor.transformers.utils.helpers import (
    detect_last_checkpoint,
    is_model_ct_quantized_from_path,
)
from llmcompressor.typing import Processor
from llmcompressor.utils.fsdp.helpers import is_fsdp_model


def train(**kwargs):
    """
    CLI entrypoint for running training
    """
    model_args, data_args, recipe_args, training_args = parse_args(**kwargs)
    training_args.do_train = True
    main(model_args, data_args, recipe_args, training_args)


def eval(**kwargs):
    """
    CLI entrypoint for running evaluation
    """
    model_args, data_args, recipe_args, training_args = parse_args(**kwargs)
    training_args.do_eval = True
    main(model_args, data_args, recipe_args, training_args)


@deprecated(
    message=(
        "`from llmcompressor.transformers import oneshot` is deprecated, "
        "please use `from llmcompressor import oneshot`."
    )
)
def oneshot(**kwargs) -> None:
    from llmcompressor import oneshot

    oneshot(**kwargs)


def apply(**kwargs):
    """
    CLI entrypoint for any of training, oneshot
    """
    report_to = kwargs.get("report_to", None)
    model_args, data_args, recipe_args, training_args = parse_args(**kwargs)

    training_args.run_stages = True
    if report_to is None:  # user didn't specify any reporters
        # get rid of the reporters inferred from hugging face
        training_args.report_to = []
    main(model_args, data_args, recipe_args, training_args)


def compress(**kwargs):
    apply(**kwargs)


def parse_args(**kwargs):
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
    parser = HfArgumentParser(
        (ModelArguments, DatasetArguments, RecipeArguments, TrainingArguments)
    )

    if not kwargs:
        parsed_args = parser.parse_args_into_dataclasses()
    else:
        parsed_args = parser.parse_dict(kwargs)

    model_args, data_args, recipe_args, training_args = parsed_args
    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    # raise depreciation warnings
    if data_args.remove_columns is not None:
        warnings.warn(
            "`remove_columns` argument is depreciated. When tokenizing datasets, all "
            "columns which are invalid inputs the tokenizer will be removed",
            DeprecationWarning,
        )

    # silently assign tokenizer to processor
    if model_args.tokenizer:
        if model_args.processor:
            raise ValueError("Cannot use both a tokenizer and processor")
        model_args.processor = model_args.tokenizer
    model_args.tokenizer = None

    return model_args, data_args, recipe_args, training_args


def initialize_model_from_path(
    model_args: ModelArguments,
    training_args: Optional[TrainingArguments] = None,
):
    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    model_path = model_args.model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        tie_word_embeddings=model_args.tie_word_embeddings,
        trust_remote_code=model_args.trust_remote_code_model,
    )

    last_checkpoint = None
    teacher = None

    if training_args is not None:
        # Load teacher configuration if applicable
        teacher_config = (
            AutoConfig.from_pretrained(
                model_args.distill_teacher,
                use_auth_token=True if model_args.use_auth_token else None,
                tie_word_embeddings=model_args.tie_word_embeddings,
                trust_remote_code=model_args.trust_remote_code_model,
            )
            if model_args.distill_teacher
            else None
        )

        # Detect last checkpoint
        last_checkpoint = detect_last_checkpoint(training_args, model_args=model_args)

        # Set seed before initializing model
        set_seed(training_args.seed)

        # Initialize teacher model if teacher path is provided
        if model_args.distill_teacher is not None:
            teacher_device_map = (
                None
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
                else "auto"
            )
            teacher_kwargs = {
                "config": teacher_config,
                "cache_dir": model_args.cache_dir,
                "use_auth_token": True if model_args.use_auth_token else None,
                "torch_dtype": parse_dtype(model_args.precision),
                "device_map": teacher_device_map,
                "trust_remote_code": model_args.trust_remote_code_model,
            }

            teacher = AutoModelForCausalLM.from_pretrained(
                model_args.distill_teacher,
                **teacher_kwargs,
            )
            if "sequence_length" in teacher_kwargs:
                teacher.seqlen = teacher_kwargs["sequence_length"]

    model_path = (
        last_checkpoint or model_args.model
        if hasattr(model_args, "model")
        else model_args.model_name_or_path
    )

    # Fallback to CPU if GPU requested and not available
    model_args.oneshot_device = fallback_to_cpu(model_args.oneshot_device)

    # Trainer handles device assignment for FSDP and training, don't do mapping here
    # if running oneshot outside of FSDP, apply user device settings

    fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"

    device_map = model_args.oneshot_device
    if not fsdp_enabled and training_args is not None and training_args.do_train:
        device_map = "auto"

    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": device_map,
        "trust_remote_code": model_args.trust_remote_code_model,
    }

    # this calls from_pretrained under the hood so should be FSDP safe

    # optimized models must be decompressed to carry out oneshot/train/etc
    if is_model_ct_quantized_from_path(model_path):
        model_kwargs["quantization_config"] = CompressedTensorsConfig(
            run_compressed=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if "sequence_length" in model_kwargs:
        model.seqlen = model_kwargs["sequence_length"]

    return model, teacher


def initialize_processor_from_path(
    model_args: ModelArguments,
    model: PreTrainedModel,
    teacher: Optional[PreTrainedModel] = None,
) -> Processor:
    processor_src = model_args.processor or get_processor_name_from_model(
        model, teacher
    )
    # The use_fast=True option is not currently supported safely in Transformers
    # See: https://github.com/huggingface/transformers/pull/34836#issuecomment-2491809727  # noqa: E501
    try:
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )
    except Exception:
        logger.debug("Could not load fast processor, loading slow processor instead")
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=model_args.cache_dir,
            use_fast=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )

    return processor


def main(
    model_args: ModelArguments,
    data_args: DatasetArguments,
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
    :param data_args: Arguments pertaining to what data we are going to input our model
    for training
    :param training_args: Arguments pertaining to training loop configuration
    """

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

    pre_initialize_structure(model=model)

    # initialize session manager
    initialize_recipe(model, None)

    # Load datasets
    stage_runner = StageRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        recipe_args=recipe_args,
    )
    add_labels = training_args.do_train or training_args.run_stages
    stage_runner.populate_datasets(processor=processor, add_labels=add_labels)
    train_dataset = stage_runner.get_dataset_split("train")
    calib_dataset = stage_runner.get_dataset_split("calibration")

    trainer = Trainer(
        model_init=get_session_model,
        teacher=teacher,
        recipe=recipe_args.recipe,
        recipe_args=recipe_args.recipe_args,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        train_dataset=train_dataset or calib_dataset,
        processing_class=processor,
        data_collator=data_args.data_collator,
    )

    # wrap model.save_pretrained
    if is_fsdp_model(model):
        modify_fsdp_model_save_pretrained(trainer, processor)
    else:
        modify_save_pretrained(model)

    stage_runner.trainer = trainer

    # alternating Training/One-shot
    if training_args.run_stages:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        stage_runner.run_sequential_stages(checkpoint)

        # exit immediately
        return
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        stage_runner.train(checkpoint)

    # save if model was provided as a string or custom output_dir was set

    if isinstance(model_args.model, str) or (
        training_args.output_dir
        != TrainingArguments.__dataclass_fields__["output_dir"].default
    ):
        model.save_pretrained(
            training_args.output_dir, save_compressed=model_args.save_compressed
        )
        if processor is not None:
            processor.save_pretrained(training_args.output_dir)

    # Clean up the CompressionSession before exit if requested
    if recipe_args.clear_sparse_session:
        reset_session()


if __name__ == "__main__":
    apply()
