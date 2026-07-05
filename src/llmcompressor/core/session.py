"""
Compression session management for LLM compression workflows.

Provides the main CompressionSession class for managing compression
workflows, including lifecycle management, event handling, callback
registration, and state tracking.
"""

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from llmcompressor.core.events import EventType
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.core.state import ModifiedState, State
from llmcompressor.recipe import Recipe

__all__ = [
    "CompressionSession",
]


@dataclass
class _CallbackContainer:
    """
    A container for a callback and its deregister function

    :param id_: the id of the callback
    :param callback: the callback to invoke
    :param deregister: the function to call to deregister the callback
    :param event_type: the event type the callback is registered for
    :param kwargs: the kwargs the callback was registered with
    """

    id_: int
    callback: Callable
    deregister: Callable
    event_type: EventType
    kwargs: dict


class CompressionSession:
    """
    A session for compression that holds the lifecycle
    and state for the current compression session
    """

    def __init__(self):
        self._lifecycle = CompressionLifecycle()

    @property
    def lifecycle(self) -> CompressionLifecycle:
        """
        Lifecycle is used to keep track of where we are in the compression
        process and what modifiers are active. It also provides the ability
        to invoke events on the lifecycle.

        :return: the lifecycle for the session
        """
        return self._lifecycle

    @property
    def state(self) -> State:
        """
        State of the current compression session. State instance
        is used to store all information such as the recipe, model
        optimizer, data, etc. that is needed for compression.

        :return: the current state of the session
        """
        return self._lifecycle.state

    def initialize(
        self,
        recipe: str | list[str] | Recipe | list[Recipe] | None = None,
        recipe_stage: str | list[str] | None = None,
        recipe_args: dict[str, Any] | None = None,
        model: Any | None = None,
        teacher_model: Any | None = None,
        optimizer: Any | None = None,
        attach_optim_callbacks: bool = True,
        train_data: Any | None = None,
        val_data: Any | None = None,
        test_data: Any | None = None,
        calib_data: Any | None = None,
        copy_data: bool = True,
        start: float | None = None,
        steps_per_epoch: int | None = None,
        batches_per_step: int | None = None,
        **kwargs,
    ) -> ModifiedState:
        """
        Initialize the session for compression. This will run the initialize method
        for each modifier in the session's lifecycle. This will also set the session's
        state to the initialized state.

        :param recipe: the recipe to use for the compression, can be a path to a
            recipe file, a raw recipe string, a recipe object, or a list
            of recipe objects.
        :param recipe_stage: the stage to target for the compression
        :param recipe_args: the args to use for overriding the recipe defaults
        :param model: the model to compress
        :param teacher_model: the teacher model to use for knowledge distillation
        :param optimizer: the optimizer to use for the compression
        :param attach_optim_callbacks: True to attach the optimizer callbacks to the
            compression lifecycle, False otherwise
        :param train_data: the training data to use for the compression
        :param val_data: the validation data to use for the compression
        :param test_data: the testing data to use for the compression
        :param calib_data: the calibration data to use for the compression
        :param copy_data: True to copy the data, False otherwise
        :param start: the start epoch to use for the compression
        :param steps_per_epoch: the number of steps per epoch to use for the
            compression
        :param batches_per_step: the number of batches per step to use for
            compression
        :param kwargs: additional kwargs to pass to the lifecycle's initialize method
        :return: the modified state of the session after initializing
        """
        mod_data = self._lifecycle.initialize(
            recipe=recipe,
            recipe_stage=recipe_stage,
            recipe_args=recipe_args,
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            attach_optim_callbacks=attach_optim_callbacks,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            calib_data=calib_data,
            copy_data=copy_data,
            start=start,
            steps_per_epoch=steps_per_epoch,
            batches_per_step=batches_per_step,
            **kwargs,
        )

        return ModifiedState(
            model=self.state.model,
            optimizer=self.state.optimizer,
            loss=self.state.loss,
            modifier_data=mod_data,
        )

    def finalize(self, **kwargs) -> ModifiedState:
        """
        Finalize the session for compression. This will run the finalize method
        for each modifier in the session's lifecycle. This will also set the session's
        state to the finalized state.

        :param kwargs: additional kwargs to pass to the lifecycle's finalize method
        :return: the modified state of the session after finalizing
        """
        mod_data = self._lifecycle.finalize(**kwargs)

        return ModifiedState(
            model=self.state.model,
            optimizer=self.state.optimizer,
            loss=self.state.loss,
            modifier_data=mod_data,
        )

    def event(
        self,
        event_type: EventType,
        batch_data: Any | None = None,
        loss: Any | None = None,
        **kwargs,
    ) -> ModifiedState:
        """
        Invoke an event for current CompressionSession.

        :param event_type: the event type to invoke
        :param batch_data: the batch data to use for the event
        :param loss: the loss to use for the event if any
        :param kwargs: additional kwargs to pass to the lifecycle's event method
        :return: the modified state of the session after invoking the event
        """
        mod_data = self._lifecycle.event(
            event_type=event_type, batch_data=batch_data, loss=loss, **kwargs
        )
        return ModifiedState(
            model=self.state.model,
            optimizer=self.state.optimizer,
            loss=self.state.loss,  # TODO: is this supposed to be a different type?
            modifier_data=mod_data,
        )

    def reset(self):
        """
        Reset the session to its initial state
        """
        self._lifecycle.reset()

    def reset_stage(self):
        """
        Reset the session for starting a new stage, recipe and model stays intact
        """
        self.lifecycle.initialized_ = False
        self.lifecycle.finalized = False

    def get_serialized_recipe(self) -> str | None:
        """
        :return: serialized string of the current compiled recipe
        """
        recipe = self.lifecycle.recipe

        if recipe is not None and hasattr(recipe, "yaml"):
            return recipe.yaml()

        logger.warning("Recipe not found in session - it may have been reset")
