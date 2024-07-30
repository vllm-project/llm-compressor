import inspect
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from llmcompressor.core import (
    active_session,
    apply,
    callbacks,
    create_session,
    finalize,
    initialize,
    pre_initialize_structure,
)
from llmcompressor.metrics import LoggerManager
from llmcompressor.pytorch.model_load.helpers import RECIPE_FILE_NAME, get_session_model
from llmcompressor.pytorch.utils import ModuleSparsificationInfo
from llmcompressor.transformers.finetune.callbacks import (
    DisableHalfPrecisionCallback,
    TrainingLoopCallbacks,
)
from llmcompressor.utils.fsdp.context import summon_full_params_context
from llmcompressor.utils.fsdp.helpers import is_fsdp_model, save_pretrained_fsdp
from llmcompressor.utils.pytorch import qat_active

__all__ = [
    "SessionManagerMixIn",
]

TRAINER_STATE_NAME = "trainer_state.json"
METADATA_ARGS = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "max_seq_length",
    "save_safetensors",
    "fp16",
]


class SessionManagerMixIn:
    """
    Mix-In class to extend the Hugging Face Trainer class to support LLM Compressor
    recipes for one-shot and finetuning flows.

    :param recipe: path to recipe file to apply during training
    :param recipe_args: additional kwargs to use for evaluating recipe
    :param data_args: kwargs for configuring dataset loading
    :param teacher: optional teacher model to use for distillation
    """

    def __init__(
        self,
        recipe: Optional[str] = None,
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        data_args: Optional["DataTrainingArguments"] = None,  # noqa: F821
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        self.recipe = recipe
        self.recipe_args = recipe_args
        self.teacher = teacher

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
        create_session()

        # call Trainer initialization
        super().__init__(**kwargs)
        self.accelerator.wait_for_everyone()

        # setup callbacks and loss
        self.optim_callbacks = TrainingLoopCallbacks(self)
        self.callback_handler.add_callback(self.optim_callbacks)
        self.callback_disable_fp16 = DisableHalfPrecisionCallback(self)
        self.callback_handler.add_callback(self.callback_disable_fp16)
        self.criterion = torch.nn.CrossEntropyLoss()

        model_signature = inspect.signature(self.model.forward)
        self._signature_columns = list(model_signature.parameters.keys())

        if self.teacher is not None and teacher not in ("disable", "self"):
            teacher_signature = inspect.signature(self.teacher.forward)
            self._teacher_signature_columns = list(teacher_signature.parameters.keys())
        else:
            self._teacher_signature_columns = None

        if self.is_fsdp_enabled:
            self._prepare_model_for_fsdp()

        if data_args is not None:
            self.min_tokens_per_module = data_args.min_tokens_per_module

    def initialize_session(
        self,
        epoch: float,
        checkpoint: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """
        Initialize the CompressionSession from the specified epoch, evaluates the recipe
        and initialized the modifiers for the training session

        :param epoch: Epoch to initialize session from, usually 0 unless loading
        from a checkpoint
        :param checkpoint: Optional checkpoint to initialize from to continue training
        :param stage: Optional stage of recipe to run, or None to run all stages
        """
        session = active_session()
        if session.lifecycle.initialized_ or session.lifecycle.finalized:
            return False

        train_data = self.get_train_dataloader()

        self.accelerator.wait_for_everyone()
        with summon_full_params_context(self.model, offload_to_cpu=True):
            initialize(
                model=self.model,
                teacher_model=self.teacher,  # TODO: what about for self/disable?
                recipe=self.recipe,
                recipe_stage=stage,
                recipe_args=self.recipe_args,
                train_data=train_data,
                start=epoch,
                copy_data=False,
                fsdp_active=self.is_fsdp_enabled,
                metadata=self.metadata,
            )
        self.accelerator.wait_for_everyone()
        model = get_session_model()
        self.model_wrapped = self.model = model

        if self.recipe is None:
            logger.warning(
                "No training recipe was provided, finetuning will be run "
                "without event callbacks to LLM Compressor. To supply a recipe "
                "pass a yaml file or string to the `recipe` argument."
            )

        torch.cuda.empty_cache()

    def initialize_structure(self, stage: Optional[str] = None):
        """
        Initialize any recipe structural changes such as quantization on the model,
        return immediately if session has already been initialized

        :param stage: Optional stage of recipe to run, or None to run all stages
        """
        session = active_session()
        if session.lifecycle.initialized_:
            return False

        pre_initialize_structure(
            model=self.model,
            recipe=self.recipe,
            recipe_stage=stage,
            recipe_args=self.recipe_args,
        )
        logger.info(f"Initialized LLM Compressor structure from recipe {self.recipe}")
        torch.cuda.empty_cache()

    def finalize_session(self):
        """
        Wrap up training by finalizing all modifiers initialized in the current session
        """
        session = active_session()
        if not session.lifecycle.initialized_ or session.lifecycle.finalized:
            return False

        with summon_full_params_context(self.model, offload_to_cpu=True):
            # in order to update each layer we need to gathers all its parameters
            finalize()
        logger.info("Finalized LLM Compressor session")
        model = get_session_model()
        self.model = model
        torch.cuda.empty_cache()

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

        initialize(optimizer=self.optimizer, steps_per_epoch=self.total_steps_per_epoch)

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
        return super().create_scheduler(num_training_steps, optimizer)

    def training_step(
        self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Overrides the Trainer's training step to trigger the batch_start callback to
        the modifiers, then calls the parent function.

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :return: output of the model
        """
        self._check_super_defined("training_step")

        callbacks.batch_start(batch_data=inputs)
        model_outputs = super().training_step(model, inputs)

        return model_outputs

    def compute_loss(
        self, model: Module, inputs: Dict[str, Any], return_outputs: bool = False
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
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

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

        if active_session().lifecycle.initialized_:
            state = callbacks.loss_calculated(loss=loss)
            if state and state.loss is not None:
                loss = state.loss
                if do_log:
                    log["distill_step_loss"] = loss.item() - log["step_loss"]
            callbacks.optim_pre_step()

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
            model, inputs, prediction_loss_only, ignore_keys
        )
        return model_outputs

    def train(self, *args, stage: Optional[str] = None, **kwargs):
        """
        Run a sparsification training cycle. Runs initialization for the sparse session
        before calling super().train() and finalization of the session after.

        Logs sparsification details for the trained model.

        :param args: positional args to pass to super().train()
        :param stage: Optional stage of recipe to run, or None to run all stages
        :param kwargs: keyword args to pass to super().train()
        :return: the output from super.train()
        """
        checkpoint, epoch = self._calculate_checkpoint_info(kwargs)
        self.initialize_session(epoch=epoch, checkpoint=checkpoint, stage=stage)
        self.accelerator.wait_for_everyone()
        output = super().train(*args, **kwargs)
        self.accelerator.wait_for_everyone()
        self.finalize_session()

        self.accelerator.wait_for_everyone()

        # log model sparsity
        self.maybe_log_model_sparsification()
        self.accelerator.wait_for_everyone()

        return output

    def evaluate(self, *args, **kwargs):
        """
        Run a sparsification evaluation cycle.
        Runs initialize_structure for the sparse session before calling
        super().evaluate() and finalization of the session after.

        :param args: positional args to pass to super().evaluate()
        :param kwargs: keyword args to pass to super().evaluate()
        :return: the output from super.evaluate()
        """
        self.initialize_structure()

        output = super().evaluate(*args, **kwargs)
        self.finalize_session()

        return output

    def predict(self, *args, **kwargs):
        """
        Run a sparsification prediction cycle.
        Runs initialize_structure for the sparse session before calling
        super().predict() and finalization of the session after.

        :param args: positional args to pass to super().predict()
        :param kwargs: keyword args to pass to super().predict()
        :return: the output from super.predict()
        """
        self.initialize_structure()
        output = super().predict(*args, **kwargs)
        self.finalize_session()

        return output

    def one_shot(
        self, calibration_data: Optional[DataLoader] = None, stage: Optional[str] = None
    ):
        """
        Run oneshot calibration on the active model

        :param stage: which stage of the recipe to run, or None to run whole recipe
        :param calib_data: dataloader of calibration data
        """
        apply(
            recipe=self.recipe,
            recipe_stage=stage,
            recipe_args=self.recipe_args,
            model=self.model,
            calib_data=calibration_data,
            start=-1,
            copy_data=False,
            accelerator=self.accelerator,
            min_tokens_per_module=self.min_tokens_per_module,
        )

        # log model sparsity
        # self.maybe_log_model_sparsification()
        self.accelerator.wait_for_everyone()

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call=False, _is_oneshot=False
    ):
        """
        Override of the save_model function and expects it to exist in the parent.
        Calls into super() to save the model and additionally saves any recipes
        that were used with the model within the model folder.

        :param output_dir: the path to save the recipes into
        """
        if active_session() is None:
            return  # nothing to save

        if output_dir is None:
            output_dir = self.args.output_dir

        if not is_fsdp_model(self.model):
            self.model.save_pretrained(
                output_dir,
                save_compressed=self.args.save_compressed,
                safe_serialization=self.args.save_safetensors,
            )
        else:  # FSDP model
            save_pretrained_fsdp(
                model=self.model,
                accelerator=self.accelerator,
                output_dir=output_dir,
                save_compressed=self.args.save_compressed,
                save_safetensors=self.metadata.get("save_safetensors", False),
            )

        self.save_state()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        if not self.recipe:
            return

        if self.accelerator.is_main_process:
            # save recipe, will contain modifiers from the model's original recipe as
            # well as those added from self.recipe
            recipe_path = os.path.join(output_dir, RECIPE_FILE_NAME)
            session = active_session()
            recipe_yaml_str = session.get_serialized_recipe()
            with open(recipe_path, "w") as fp:
                fp.write(recipe_yaml_str)

            logger.info(
                f"Saved LLM Compressor recipe with model state to {recipe_path}"
            )

        self.accelerator.wait_for_everyone()

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
