import math

from transformers import TrainerCallback, TrainerControl, TrainingArguments
from transformers.trainer_callback import TrainerState

from llmcompressor.core.llmcompressor.globals import get_compressor, get_state

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
        get_state().model = self.trainer.model

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
        compressor = get_compressor()
        compressor.optim_post_step()
        compressor.batch_end()

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
        compressor = get_compressor()
        compressor.optim_post_step()
        compressor.batch_end()


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TODO: I don't think this callback actually does anything?

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
