import inspect
import math
import os
from dataclasses import asdict
from pathlib import PosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer as HFTrainer
from transformers.trainer_callback import TrainerState

from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.metrics import LoggerManager
from llmcompressor.pytorch.utils import ModuleSparsificationInfo
from llmcompressor.transformers import DataTrainingArguments
from llmcompressor.transformers.finetune.callbacks import (
    DisableHalfPrecisionCallback,
    TrainingLoopCallbacks,
)
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_calibration_dataloader,
)
from llmcompressor.transformers.finetune.model_args import ModelArguments
from llmcompressor.transformers.finetune.text_generation import (
    initialize_model_from_path,
    initialize_processor_from_path,
    parse_args,
)
from llmcompressor.transformers.finetune.training_args import DEFAULT_OUTPUT_DIR
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.utils.recipe_args import RecipeArguments
from llmcompressor.utils.fsdp.context import summon_full_params_context

__all__ = ["Train"]


TRAINER_STATE_NAME = "trainer_state.json"
METADATA_ARGS = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "max_seq_length",
    "save_safetensors",
    "fp16",
]


class Trainer(HFTrainer):
    def __init__(
        self,
        training_dataset,
        eval_dataset,
        lifecycle,
        recipe_args: Optional["RecipeArguments"] = None,
        data_args: Optional["DataTrainingArguments"] = None,
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ):
        self.recipe = recipe_args.recipe
        self.recipe_args = recipe_args.recipe_args
        self.model_args = model_args
        self.teacher = model_args.distill_teacher
        self.lifecycle = lifecycle

        self.callbacks = LifecycleCallbacks()

        # parse training and metadata args
        training_args = kwargs.get("args")
        self.metadata = (
            self._extract_metadata(
                metadata_args=METADATA_ARGS,
                training_args_dict=training_args.to_dict(),
                data_args_dict=asdict(data_args) if data_args else {},
            )
            if training_args and METADATA_ARGS
            else None
        )

        # setup metrics and session
        self.logger_manager = LoggerManager(log_python=False)

        # call Trainer initialization
        super().__init__(
            model=model_args.model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            processing_class=model_args.processor,
            data_collator=data_args.data_collator,
        )
        self.accelerator.wait_for_everyone()

        # setup callbacks and loss
        self.optim_callbacks = TrainingLoopCallbacks(
            trainer=self, callbacks=self.callbacks
        )
        self.callback_handler.add_callback(self.optim_callbacks)
        self.callback_disable_fp16 = DisableHalfPrecisionCallback(self)
        self.callback_handler.add_callback(self.callback_disable_fp16)
        self.criterion = torch.nn.CrossEntropyLoss()

        model_signature = inspect.signature(self.model.forward)
        self._signature_columns = list(model_signature.parameters.keys())

        if self.teacher is not None and self.teacher not in ("disable", "self"):
            teacher_signature = inspect.signature(self.teacher.forward)
            self._teacher_signature_columns = list(teacher_signature.parameters.keys())
        else:
            self._teacher_signature_columns = None

        if self.is_fsdp_enabled:
            self._prepare_model_for_fsdp()

        if data_args is not None:
            self.min_tokens_per_module = data_args.min_tokens_per_module

    def create_optimizer(self):
        """
        Override the optimizer to apply and update the recipe while training.
        create_optimizer must exist in the parent class and should set
        self.optimizer to the optimizer state and optionally set self.scaler
        if using amp.
        """

        self._check_super_defined("create_optimizer")
        super().create_optimizer()

        # n_gpu handled internally by dataloader
        total_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )

        if isinstance(self.train_dataset, IterableDataset):
            logger.warning(
                "Training is being run with a streamed dataset, "
                "steps_per_epoch cannot be determined and will default to "
                "1. LLM Compressor modifiers utilizing this statistic may not "
                "behave as expected. "
            )
            self.total_steps_per_epoch = 1
        else:
            self.total_steps_per_epoch = math.ceil(
                len(self.train_dataset) / total_batch_size
            )

        self.lifecycle.initialize(
            optimizer=self.optimizer, steps_per_epoch=self.total_steps_per_epoch
        )

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Create an LR scheduler to work with the applied recipes. This is a placeholder
        that just calls the super method, but would be expanded upon if we ever
        implement a LearningRateModifier.

        :param num_training_steps: the total number of training steps
        :param optimizer: pre-initialized optimizer
        """

        # TODO: we don't currently have a LR scheduler in the new modifier framework
        self._check_super_defined("create_scheduler")
        return super().create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Overrides the Trainer's training step to trigger the batch_start callback to
        the modifiers, then calls the parent function.

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :return: output of the model
        """
        self._check_super_defined("training_step")

        self.callbacks.batch_start(batch_data=inputs)
        model_outputs = super().training_step(
            model=model, inputs=inputs, num_items_in_batch=num_items_in_batch
        )

        return model_outputs

    def compute_loss(
        self,
        model: Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override for the compute_loss to factor trigger callbacks and filter columns

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :param return_outputs: True to return the outputs with the loss,
            False otherwise
        :return: the resulting loss if not return_outputs, otherwise a tuple
            containing the loss and the model's outputs
        """
        self._check_super_defined("compute_loss")

        # TODO: do we need these model signature columns?
        inputs = {k: inputs[k] for k in inputs if k in self._signature_columns}
        loss = super().compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        # take the mean across multiple GPUs
        # this is done outside the compute_loss function in the parent, replicating it
        # here for LLM Compressor logging and distillation
        loss = loss.mean()

        # Log step-wise loss and perplexity, for llama-recipes comparison
        # we want this before distillation loss so perplexity isn't thrown off
        do_log = self.state.global_step % self.args.logging_steps == 0
        if do_log:
            log = {}
            log["step_loss"] = loss.item()
            log["perplexity"] = torch.exp(loss).item()

        # if active_session().lifecycle.initialized_:
        state = self.callbacks.loss_calculated(loss=loss)
        if state and state.loss is not None:
            loss = state.loss
            if do_log:
                log["distill_step_loss"] = loss.item() - log["step_loss"]
        self.callbacks.optim_pre_step()

        if do_log:
            self.log(log)

        return loss

    def prediction_step(
        self,
        model: Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Wraps the prediction step from the original trainer to remove any input entry
        that should not be passed to the model.
        This situation may arise when distillation is used and the teacher model
        contains more inputs than the student model.
        """
        self._check_super_defined("prediction_step")

        inputs = {k: inputs[k] for k in inputs if k in self._model_signature_columns}

        model_outputs = super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        return model_outputs

    def maybe_log_model_sparsification(self):
        """
        Log info on model sparsity and quantization if possible. Only print logs on the
        main process, and avoid logging for quantized FSDP models
        """
        with summon_full_params_context(self.model, offload_to_cpu=True):
            # offload to avoid OOM errors
            if not self.accelerator.is_main_process:
                # only calculate stats rank0 GPU
                return
            if self.is_fsdp_enabled and qat_active(self.model):
                # due to state dict changes we can't log sparsity info with quantized
                # models in FSDP
                return

            self.log_model_sparsification()

    def log_model_sparsification(self):
        """
        Log the current model sparsification info including pruned and quantized states
        """
        sparsification_info = ModuleSparsificationInfo(self.model)

        logger.info(
            f"Sparsification info for {type(self.model).__name__}: "
            f"{sparsification_info.params_total} total params. "
        )
        sparsity_percent_formatted = "{:.2f}".format(
            sparsification_info.params_sparse_percent
        )
        logger.info(
            f"There are {sparsification_info.params_total} prunable "
            f"params which have {sparsity_percent_formatted}% "
            "avg sparsity."
        )

        quant_percent_formatted = "{:.2f}".format(
            sparsification_info.params_quantized_percent
        )
        logger.info(
            f"There are {sparsification_info.params_total} quantizable "
            f"params, with a quantization percentage of "
            f"{quant_percent_formatted}%."
        )

    def _prepare_model_for_fsdp(self):
        """
        Sets up FSDP ahead of time so we can run one-shot in FSDP mode
        """
        self.model.to("cpu")
        self.model = self.accelerator.prepare(self.model)
        self.accelerator.wait_for_everyone()

        if self.teacher is not None:
            self.teacher.to("cpu")
            for n, p in self.teacher.named_parameters():
                p.requires_grad = False
            self.teacher = self.accelerator.prepare(self.teacher)
            self.teacher.eval()
            self.accelerator.wait_for_everyone()

    def _extract_metadata(
        self,
        metadata_args: List[str],
        training_args_dict: Dict[str, Any],
        data_args_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = {}
        if not training_args_dict.keys().isdisjoint(data_args_dict.keys()):
            raise ValueError(
                "Found common keys in `training_args` and `data args`. "
                "This is prohibitive and may lead to undesired behavior."
            )

        args_dict = {**training_args_dict, **data_args_dict}

        for arg in metadata_args:
            if arg not in args_dict.keys():
                logger.warning(
                    f"Required metadata argument {arg} was not found "
                    f"in the training arguments. Setting {arg} to None."
                )
                metadata[arg] = None
            else:
                metadata[arg] = args_dict[arg]

        return metadata

    def _check_super_defined(self, func: str):
        if not hasattr(super(), func):
            raise NotImplementedError(
                f"The super class for SessionManagerMixIn must define a {func} function"
            )

    def _calculate_checkpoint_info(self, kwargs) -> Tuple[Optional[str], float]:
        """
        If resuming from checkpoint is set, get checkpoint and epoch to resume from
        """
        checkpoint = None
        epoch = 0.0

        if not kwargs or "resume_from_checkpoint" not in kwargs:
            logger.warning(
                "resume_from_checkpoint not passed into LLM Compressor Trainer.train. "
                "This will cause issues with restoring recipes when "
                "running from a checkpoint."
            )
        elif kwargs["resume_from_checkpoint"]:
            if (
                isinstance(kwargs["resume_from_checkpoint"], bool)
                and kwargs["resume_from_checkpoint"]
            ):
                checkpoint = get_last_checkpoint(self.args.output_dir)
            else:
                checkpoint = kwargs["resume_from_checkpoint"]
            epoch = TrainerState.load_from_json(
                os.path.join(checkpoint, TRAINER_STATE_NAME)
            ).epoch

        return checkpoint, epoch


class Train:
    """
    Class responsible for carrying out oneshot calibration.

    Usage:

    ```python
    trainer = Train(model=model, recipe=recipe, dataset=dataset)
    trainer.run()

    model = trainer.model
    tokenizer_or_processor = trainer.tokenizer_or_processor
    recipe = trainer.recipe

    ```
    """

    MODIFIER_LIFECYCLE_ACTIONS = (
        "initialize",
        "finalize",
    )

    def __init__(self, **kwargs):
        self.model_args, self.data_args, self.recipe_args, self.training_args = (
            parse_args(**kwargs)
        )

        self.lifecycle = CompressionLifecycle()
        self.output_dir = self.training_args.output_dir
        self.checkpoint = None

        # Preprocess the model and tokenizer/processor
        self._pre_process()

        training_dataset, eval_dataset = get_calibration_dataloader(
            self.data_args,
            self.model_args.processor,
            add_labels=True,
            do_oneshot=False,
            do_train=True,
        )

        self.trainer = Trainer(
            model_args=self.model_args,
            data_args=self.data_args,
            recipe_args=self.recipe_args,
            training_dataset=training_dataset,
            eval_dataset=eval_dataset,
            lifecycle=self.lifecycle,
        )

        # Set instance attributes
        self.model = self.model_args.model
        self.teacher = self.model_args.distill_teacher
        self.tokenizer_or_processor = self.model_args.processor
        self.recipe = self.recipe_args.recipe
        self.modifiers = self.lifecycle.modifiers

    def run(self):
        ############ initialize  ##################
        train_data = self.trainer.get_train_dataloader()

        self.trainer.accelerator.wait_for_everyone()
        _, epoch = self._calculate_checkpoint_info(
            resume_from_checkpoint=self.checkpoint
        )

        with summon_full_params_context(self.model, offload_to_cpu=True):
            self.lifecycle.initialize(
                model=self.model,
                teacher_model=self.teacher,
                recipe=self.recipe,
                recipe_args=self.recipe_args,
                train_data=train_data,
                start=epoch,
                copy_data=False,
                fsdp_active=self.trainer.is_fsdp_enabled,
                metadata=self.trainer.metadata,
            )

        self.trainer.accelerator.wait_for_everyone()
        self.model_wrapped = self.model

        if self.recipe is None:
            logger.warning(
                "No training recipe was provided, finetuning will be run "
                "without event callbacks to LLM Compressor. To supply a recipe "
                "pass a yaml file or string to the `recipe` argument."
            )

        torch.cuda.empty_cache()

        self.trainer.accelerator.wait_for_everyone()
        output = self.trainer.train()
        self.trainer.accelerator.wait_for_everyone()

        ############ finalize  ##################
        self.trainer.accelerator.wait_for_everyone()
        with summon_full_params_context(self.model, offload_to_cpu=True):
            self.lifecycle.finalize()

        torch.cuda.empty_cache()
        self.trainer.accelerator.wait_for_everyone()

        self._post_process()
        return output

    def save(self):
        """Save the model and tokenizer/processor to the output directory"""
        self.model.save_pretrained(
            self.output_dir,
            save_compressed=self.model_args.save_compressed,
            stage_modifiers=self.lifecycle.modifiers,
        )
        if self.tokenizer_or_processor:
            self.tokenizer_or_processor.save_pretrained(self.output_dir)

    def _apply_recipe_modifiers(self, calibration_dataloader: Optional[DataLoader]):
        """Apply recipe modifiers to the model"""
        for action in self.MODIFIER_LIFECYCLE_ACTIONS:
            lifecycle = getattr(self.lifecycle, action)
            lifecycle(
                model=self.model,
                recipe=self.recipe,
                recipe_args=self.recipe_args.recipe_args,
                calib_data=calibration_dataloader,
                start=-1,  # oneshot-specific argument
                copy_data=False,
                min_tokens_per_module=getattr(self, "min_tokens_per_module", None),
            )

    def _pre_process(self):
        """Preprocess model and tokenizer/processor"""
        self._warn_tied_embeddings()

        # Initialize model
        if isinstance(self.model_args.model, (str, PosixPath)):
            self.model_args.model, self.model_args.distill_teacher = (
                initialize_model_from_path(self.model_args)
            )

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

        if self.training_args.resume_from_checkpoint is not None:
            self.checkpoint = self.training_args.resume_from_checkpoint

    def _warn_tied_embeddings(self):
        if self.model_args.tie_word_embeddings:
            logger.debug(
                "The tie_word_embeddings flag is by default set to False. "
                "This guarantees that the one-shot algorithm saves the final "
                "weights without errors. Detected tie_word_embeddings=True. "
                "This may cause issues with the one-shot algorithm on save"
            )

    def _post_process(self):
        """Save model and reset the lifecycle if requested"""
        if (
            isinstance(self.model_args.model, str)
            or self.output_dir != DEFAULT_OUTPUT_DIR
        ):
            self.save()

    def _calculate_checkpoint_info(self, **kwargs) -> Tuple[Optional[str], float]:
        """
        If resuming from checkpoint is set, get checkpoint and epoch to resume from
        """
        checkpoint = None
        epoch = 0.0

        if not kwargs or "resume_from_checkpoint" not in kwargs:
            logger.warning(
                "resume_from_checkpoint not passed into LLM Compressor Trainer.train. "
                "This will cause issues with restoring recipes when "
                "running from a checkpoint."
            )
        elif kwargs["resume_from_checkpoint"]:
            if (
                isinstance(kwargs["resume_from_checkpoint"], bool)
                and kwargs["resume_from_checkpoint"]
            ):
                checkpoint = get_last_checkpoint(self.args.output_dir)
            else:
                checkpoint = kwargs["resume_from_checkpoint"]
            epoch = TrainerState.load_from_json(
                os.path.join(checkpoint, TRAINER_STATE_NAME)
            ).epoch

        return checkpoint, epoch
