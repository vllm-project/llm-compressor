"""
Training callbacks for compression-aware fine-tuning workflows.

This module provides custom trainer callbacks that integrate compression
session management with HuggingFace training loops. Handles precision
control, training loop monitoring, and compression lifecycle events
during model fine-tuning.
"""

import math

from transformers import TrainerCallback, TrainerControl, TrainingArguments
from transformers.trainer_callback import TrainerState

from llmcompressor.core import active_session
from llmcompressor.core import callbacks as session_callbacks

__all__ = [
    "DisableHalfPrecisionCallback",
    "TrainingLoopCallbacks",
]


class TrainingLoopCallbacks(TrainerCallback):
    """
    TrainerCallback for triggering CompressionSession callbacks in the training loop.
    Used to update the model reference(for running with FSDP) and trigger the post-
    optim callbacks in each modifier.

    :param trainer: LLM Compressor trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training. Update the session reference to the
        model, as it will have changed to a wrapper if FSDP is enabled
        """
        super().on_train_begin(args, state, control, **kwargs)
        session = active_session()
        session.state.model = self.trainer.model

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation,
        one training step might take several inputs.

        Triggers optimizer post_step and batch_end in the active CompressionSession
        """
        super().on_step_end(args, state, control, **kwargs)
        session_callbacks.optim_post_step()
        session_callbacks.batch_end()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an substep during gradient accumulation.

        Triggers optimizer post_step and batch_end in the active CompressionSession
        """
        super().on_substep_end(args, state, control, **kwargs)
        session_callbacks.optim_post_step()
        session_callbacks.batch_end()


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TrainerCallback for disabling FP16 training before QAT training begins

    :param trainer: LLM Compressor trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.on_begin_called = False
        self.quant_start_epoch = math.inf

    def qat_active(self) -> bool:
        """
        :return: True if a quantization modifier is active in the current session
        """
        session = active_session()
        return session.state.model.qat_active()

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch.
        """
        super().on_epoch_begin(args, state, control, **kwargs)
        self.on_begin_called = True
