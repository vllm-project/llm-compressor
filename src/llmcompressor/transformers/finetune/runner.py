import math
import os
import re
from typing import List, Optional

import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedModel

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.core import active_session
from llmcompressor.pytorch.model_load.helpers import (
    get_completed_stages,
    save_checkpoint,
    save_completed_stages,
)
from llmcompressor.recipe import Recipe, StageRunType
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor


class StageRunner:
    """
    Launcher class for train, and one_shot flows. Manages data splits for each
    flow and configurations. In the future this class will also handle alternating
    between the different flows

    LifeCycle
        - populate_datasets()
        - set_trainer()
        - train()

    :param model_args: Arguments pertaining to model/config/processor
    :param dataset_args: Arguments pertaining to what data to use for different flows
    :param training_args: Arguments pertaining to training loop configuration
    :model: unwrapped model to run flows on
    """

    def __init__(
        self,
        dataset_args: "DatasetArguments",
        model_args: "ModelArguments",
        training_args: "TrainingArguments",
        recipe_args: "RecipeArguments",
    ):
        self._dataset_args = dataset_args
        self._model_args = model_args
        self._training_args = training_args
        self._recipe_args = recipe_args

        self.datasets = {}
        self.trainer = None
        self.processor = None
        self.parent_output_dir = self._training_args.output_dir
        self._output_dir = self._training_args.output_dir

    def populate_datasets(self, processor: Processor, add_labels: bool = True):
        """
        Loads datasets for each flow based on dataset_args, stores a Dataset for each
        enabled flow in self.datasets

        :param processor: processor or tokenizer to use for dataset tokenization
        :param add_labels: if True, add labels column to dataset splits
        """
        if self._dataset_args.dataset is None:
            self.processor = self._model_args.processor
            logger.info(
                "Running oneshot without calibration data. This is expected for "
                "weight-only and dynamic quantization"
            )
            return

        splits = self._dataset_args.splits
        tokenized_datasets = {}

        def _get_split_name(inp_str):
            # strip out split name, for ex train[60%:] -> train
            match = re.match(r"(\w*)\[.*\]", inp_str)
            if match is not None:
                return match.group(1)
            return inp_str

        if splits is None:
            splits = {"all": None}
        elif isinstance(splits, str):
            splits = {_get_split_name(splits): splits}
        elif isinstance(splits, List):
            splits = {_get_split_name(s): s for s in splits}

        # default to custom dataset if dataset provided isn't a string
        registry_id = (
            self._dataset_args.dataset
            if isinstance(self._dataset_args.dataset, str)
            else "custom"
        )
        for split_name, split_str in splits.items():
            dataset = self._dataset_args.dataset
            if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
                # dataset is already tokenized
                tokenized_datasets[split_name] = dataset
            else:
                # dataset needs to be tokenized
                dataset_manager = TextGenerationDataset.load_from_registry(
                    registry_id,
                    dataset_args=self._dataset_args,
                    split=split_str,
                    processor=processor,
                )
                tokenized_datasets[split_name] = dataset_manager(add_labels=add_labels)

        from llmcompressor.datasets import make_dataset_splits

        self.datasets = make_dataset_splits(
            tokenized_datasets,
            do_train=self._training_args.do_train,
            do_oneshot=self._training_args.do_oneshot,
        )

    def get_dataset_split(self, split_name: str) -> Dataset:
        """
        Retrieve a dataset split by name

        :param split_name: name of dataset split to return
        :return: dataset split labeled by split_name
        """
        return self.datasets.get(split_name, None)

    def train(self, checkpoint: str, stage: Optional[str] = None):
        """
        Run trainer's training loop on train_dataset, saving the resulting model to
        output_dir

        :param checkpoint: Optional checkpoint to resume from
        :param stage: which stage of the recipe to run, or None to run whole recipe
        """
        logger.info("*** Train ***")
        train_result = self.trainer.train(
            resume_from_checkpoint=checkpoint, stage=stage
        )
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.get_dataset_split("train"))
        metrics["perplexity"] = math.exp(metrics["train_loss"])
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # this includes saving the state, optimizer and scheduler
        self.trainer.save_model(output_dir=self._output_dir)

    def run_sequential_stages(
        self, model: PreTrainedModel, checkpoint: Optional[str] = None
    ):
        """
        Run the recipe stage by stage, allowing for alternating between one-shot and
        finetuning flows. Optionally save the model output at the end of each stage

        :param checkpoint: optional checkpoint to pick up a stage from
        """

        recipe_obj = Recipe.create_instance(self._recipe_args.recipe)
        with self.trainer.accelerator.main_process_first():
            checkpoint_dir = self._model_args.model
            completed_stages = get_completed_stages(checkpoint_dir)

        self.trainer.accelerator.wait_for_everyone()
        do_preprocess = True

        for stage in recipe_obj.stages:
            # validate stage
            stage_name = stage.group
            run_type = stage.infer_run_type()
            if not run_type:
                raise ValueError(
                    f"a valid stage type ({[e.value for e in StageRunType]}) "
                    "must be provided in run_stages mode. Either add a run_type "
                    "attribute to each stage in the recipe or include it as part of "
                    "the stage name."
                )

            # skip stages which have already been applied
            if stage_name in completed_stages:
                continue

            # setup checkpoint dir, TODO: this should be optional
            self._output_dir = os.path.join(
                self.parent_output_dir, "stage_" + stage_name
            )
            with self.trainer.accelerator.main_process_first():
                if not os.path.exists(self._output_dir):
                    os.makedirs(self._output_dir)
                save_completed_stages(self._output_dir, completed_stages)
            self._training_args.output_dir = self._output_dir

            # run stage
            if run_type is StageRunType.ONESHOT:
                from llmcompressor import Oneshot
                from llmcompressor.datasets import format_calibration_data

                self._model_args.model = model

                oneshot = Oneshot.from_args(
                    model_args=self._model_args,
                    dataset_args=self._dataset_args,
                    recipe_args=self._recipe_args,
                    output_dir=self._training_args.output_dir,
                    do_preprocess=do_preprocess,
                )

                calib_data = format_calibration_data(
                    tokenized_dataset=self.get_dataset_split("calibration"),
                    num_calibration_samples=self._dataset_args.num_calibration_samples,
                    do_shuffle=self._dataset_args.shuffle_calibration_samples,
                    collate_fn=self._dataset_args.data_collator,
                )

                if do_preprocess:
                    do_preprocess = False
                oneshot.apply_recipe_modifiers(
                    calibration_dataloader=calib_data,
                    recipe_stage=stage_name,
                )
            elif run_type is StageRunType.TRAIN:
                self.trainer.model = model
                self.train(checkpoint=checkpoint, stage=stage_name)

            checkpoint = None

            # save model between stages
            if (
                self._training_args.output_dir
                != TrainingArguments.__dataclass_fields__["output_dir"].default
                and self.trainer.accelerator.is_main_process
            ):
                save_checkpoint(
                    save_path=self._output_dir,
                    model=self.trainer.model,
                    processor=self.processor,
                    save_safetensors=self._training_args.save_safetensors,
                    save_compressed=self._model_args.save_compressed,
                )
            self.trainer.accelerator.wait_for_everyone()

            # save stage to checkpoint dir
            if self.trainer.accelerator.is_main_process:
                completed_stages.append(stage_name)
                save_completed_stages(self._output_dir, completed_stages)

            # setup for next stage
            session = active_session()
            session.reset()

            # synchronize and clean up memory
            self.trainer.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            self.trainer.accelerator.free_memory()
            self.trainer.accelerator.wait_for_everyone()
