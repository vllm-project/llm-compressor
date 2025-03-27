import inspect
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger
from torch.nn import Module
from transformers import PreTrainedModel
from transformers import Trainer as HFTransformersTrainer
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from llmcompressor.args.training_arguments import TrainingArguments
from llmcompressor.core.llmcompressor.globals import (
    get_compressor,
    get_model,
    get_state,
)
from llmcompressor.metrics import LoggerManager
from llmcompressor.modifiers.distillation.utils.pytorch.model_wrapper import (
    KDModelWrapper,
)
from llmcompressor.pytorch.model_load.helpers import save_checkpoint
from llmcompressor.pytorch.utils import ModuleSparsificationInfo
from llmcompressor.transformers.finetune.callbacks import (
    DisableHalfPrecisionCallback,
    TrainingLoopCallbacks,
)
from llmcompressor.typing import DatasetType, Processor
from llmcompressor.utils.fsdp.context import summon_full_params_context
from llmcompressor.utils.pytorch import qat_active

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator

__all__ = [
    "SessionManagerMixIn",
]

TRAINER_STATE_NAME = "trainer_state.json"
METADATA_ARGS = [
    "per_device_train_batch_size",
    "max_seq_length",
    "save_safetensors",
    "fp16",
]


class SessionManagerMixIn(HFTransformersTrainer):
    """
    Mix-In class to extend the Hugging Face Trainer and TRL Trainer classes to support
    LLM Compressor recipes for finetuning pathways

    Must use epoch as current_index
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[DatasetType] = None,
        eval_dataset: Optional[DatasetType] = None,
        processing_class: Optional[Processor] = None,
        **kwargs,
    ):
        # setup metrics and session
        self.logger_manager = LoggerManager(log_python=False)

        # call Trainer initialization
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.accelerator.wait_for_everyone()

        # setup callbacks and loss
        self.optim_callbacks = TrainingLoopCallbacks(self)
        self.callback_handler.add_callback(self.optim_callbacks)
        self.callback_disable_fp16 = DisableHalfPrecisionCallback(self)
        self.callback_handler.add_callback(self.callback_disable_fp16)
        self.criterion = torch.nn.CrossEntropyLoss()

        # used when computing loss
        model = get_model()
        model_signature = inspect.signature(model.forward)
        self._signature_columns = list(model_signature.parameters.keys())

        # used to toggle compressed saving

        if self.is_fsdp_enabled:
            self._prepare_model_for_fsdp()

    def initialize_session(self, epoch: float):
        """
        Initialize the CompressionSession from the specified epoch, evaluates the recipe
        and initialized the modifiers for the training session

        :param epoch: Epoch to initialize session from, usually 0 unless loading
        from a checkpoint
        :param checkpoint: Optional checkpoint to initialize from to continue training
        :param stage: Optional stage of recipe to run, or None to run all stages
        """
        compressor = get_compressor()

        self.accelerator.wait_for_everyone()
        with summon_full_params_context(self.model, offload_to_cpu=True):
            compressor.initialize(model=self.model, start=epoch)
        self.accelerator.wait_for_everyone()

    def finalize_session(self):
        """
        Wrap up training by finalizing all modifiers initialized in the current session
        """
        compressor = get_compressor()

        self.accelerator.wait_for_everyone()
        with summon_full_params_context(self.model, offload_to_cpu=True):
            compressor.finalize()
        self.accelerator.wait_for_everyone()

    def create_optimizer(self):
        """
        Override the optimizer to apply and update the recipe while training.
        create_optimizer must exist in the parent class and should set
        self.optimizer to the optimizer state and optionally set self.scaler
        if using amp.
        """
        self._check_super_defined("create_optimizer")

        super().create_optimizer()
        get_state().optimizer = self.optimizer
        return self.optimizer

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        """
        Overrides the Trainer's training step to trigger the batch_start callback to
        the modifiers, then calls the parent function.

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :return: output of the model
        """
        self._check_super_defined("training_step")

        get_compressor().batch_start(global_step=self.state.epoch)
        return super().training_step(*args, **kwargs)

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
        compressor = get_compressor()
        state = get_state()

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

        compressor.loss_calculated(loss=loss)
        if state and state.loss is not None:
            loss = state.loss
            if do_log:
                log["distill_step_loss"] = loss.item() - log["step_loss"]
        compressor.optim_pre_step()

        if do_log:
            self.log(log)

        return loss

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

        # initialize session with correct epoch
        _, epoch = self._calculate_checkpoint_info(kwargs)
        self.initialize_session(epoch=epoch)

        # train with accelerator
        self.accelerator.wait_for_everyone()
        output = super().train(*args, **kwargs)
        self.accelerator.wait_for_everyone()

        # finalize session
        self.finalize_session()
        self.accelerator.wait_for_everyone()

        # log model sparsity
        self.maybe_log_model_sparsification()
        self.accelerator.wait_for_everyone()

        return output

    def save_model(self, output_dir: str, _internal_call=False, _is_oneshot=False):
        """
        Only saves checkpoints

        Override of the save_model function and expects it to exist in the parent.
        Calls into super() to save the model and additionally saves any recipes
        that were used with the model within the model folder.

        :param output_dir: the path to save the recipes into
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # knowledge distillation requires making wrappers transparent during
        if isinstance(self.model, KDModelWrapper):
            self.model.prepare_for_save()  # TODO: move to finalize

        # save checkpoint
        self.save_state()
        if self.accelerator.is_main_process:
            processor = getattr(self, "processing_class", self.tokenizer)
            save_checkpoint(
                output_dir,
                model=self.model,
                processor=processor,
                save_safetensors=self.args.save_safetensors,
                save_compressed=False,  # Saving as a checkpoint, not final model
            )
        self.accelerator.wait_for_everyone()

        if isinstance(self.model, KDModelWrapper):
            self.model.finish_save()

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
        state = get_state()
        model = state.model
        teacher = state.teacher_model

        model = model.to("cpu")
        model = self.accelerator.prepare(model)
        self.accelerator.wait_for_everyone()

        if teacher is not None:
            teacher.to("cpu")
            for n, p in self.teacher.named_parameters():
                p.requires_grad = False
            teacher = self.accelerator.prepare(teacher)
            teacher.eval()
            self.accelerator.wait_for_everyone()

        state.model = model
        state.teacher_model = teacher

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
