"""
Module for defining and managing events in the LLM Compressor.

This module provides an Enum for different event types and a class for creating and
managing events, including methods for calculating event properties and triggering
updates based on specified intervals.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from loguru import logger

__all__ = [
    "EventType",
    "Event",
]


@unique
class EventType(Enum):
    """
    An Enum for defining the different types of events that can be triggered
    during model compression lifecycles.
    The purpose of each EventType is to trigger the corresponding
    modifier callback during training or post training pipelines.

    :param PRE_INIT: Event type for pre-initialization.
    :param INITIALIZE: Event type for initialization.
    :param FINALIZE: Event type for finalization.
    :param BATCH_START: Event type for the start of a batch.
    :param LOSS_CALCULATED: Event type for when loss is calculated.
    :param BATCH_END: Event type for the end of a batch.
    :param OPTIM_PRE_STEP: Event type for pre-optimization step.
    :param OPTIM_POST_STEP: Event type for post-optimization step.
    """

    # training lifecycle
    PRE_INIT = "pre_init"
    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # batch lifecycle
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    BATCH_END = "batch_end"

    # step lifecycle
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"

    def order(self) -> int:
        """
        Returns the priority order of the current EventType.
        Lower values have higher priority.

        :raises ValueError: if the event type is invalid.
        :return: The order of the event type, lower has higher priority.
        :rtype: int
        """
        if self == EventType.PRE_INIT:
            return 0
        elif self == EventType.INITIALIZE:
            return 10
        elif self == EventType.FINALIZE:
            return 20
        elif self == EventType.BATCH_START:
            return 100
        elif self == EventType.LOSS_CALCULATED:
            return 110
        elif self == EventType.OPTIM_PRE_STEP:
            return 120
        elif self == EventType.OPTIM_POST_STEP:
            return 130
        elif self == EventType.BATCH_END:
            return 140
        else:
            logger.error("Invalid event type: {}", self)
            raise ValueError(f"Invalid event type {self}")


@dataclass
class Event:
    """
    A class for defining an event that can be triggered during sparsification.

    :param type_: The type of event.
    :type type_: Optional[EventType]
    :param steps_per_epoch: The number of steps per epoch.
    :type steps_per_epoch: Optional[int]
    :param batches_per_step: The number of batches per step where step is an
        optimizer step invocation. For most pathways, these are the same.
        See the invocations_per_step parameter for more details when they are not.
    :type batches_per_step: Optional[int]
    :param invocations_per_step: The number of invocations of the step wrapper
        before optimizer.step was called. Generally can be left as 1 (default).
        For older amp pathways, this is the number of times the scaler wrapper
        was invoked before the wrapped optimizer step function was called to
        handle accumulation in fp16.
    :type invocations_per_step: Optional[int]
    :param global_step: The current global step.
    :type global_step: int
    :param global_batch: The current global batch.
    :type global_batch: int
    """

    type_: Optional[EventType] = None
    steps_per_epoch: Optional[int] = None
    batches_per_step: Optional[int] = None
    invocations_per_step: int = 1
    global_step: int = 0
    global_batch: int = 0

    @property
    def epoch_based(self) -> bool:
        """
        Determines if the event is based on epochs.

        :return: True if the event is based on epochs, False otherwise.
        :rtype: bool
        """
        return self.steps_per_epoch is not None

    @property
    def epoch(self) -> int:
        """
        Calculates the current epoch.

        :raises ValueError: if the event is not epoch based.
        :return: The current epoch.
        :rtype: int
        """
        if not self.epoch_based:
            logger.error("Attempt to access epoch for a non-epoch based event")
            raise ValueError("Event is not epoch based")
        return self.global_step // self.steps_per_epoch

    @property
    def epoch_full(self) -> float:
        """
        Calculates the current epoch with the fraction of the current step.

        :raises ValueError: if the event is not epoch based.
        :return: The current epoch with the fraction of the current step.
        :rtype: float
        """
        if not self.epoch_based:
            logger.error("Attempt to access epoch_full for a non-epoch based event")
            raise ValueError("Event is not epoch based")
        return self.global_step / float(self.steps_per_epoch)

    @property
    def epoch_step(self) -> int:
        """
        Calculates the current step within the current epoch.

        :raises ValueError: if the event is not epoch based.
        :return: The current step within the current epoch.
        :rtype: int
        """
        if not self.epoch_based:
            logger.error("Attempt to access epoch_step for a non-epoch based event")
            raise ValueError("Event is not epoch based")
        return self.global_step % self.steps_per_epoch

    @property
    def epoch_batch(self) -> int:
        """
        Calculates the current batch within the current epoch.

        :raises ValueError: if the event is not epoch based.
        :return: The current batch within the current epoch.
        :rtype: int
        """
        if not self.epoch_based:
            logger.error("Attempt to access epoch_batch for a non-epoch based event")
            raise ValueError("Event is not epoch based")
        batches_per_epoch = (
            self.steps_per_epoch * self.batches_per_step
            if self.batches_per_step
            else self.steps_per_epoch
        )
        return self.global_batch % batches_per_epoch

    @property
    def current_index(self) -> float:
        """
        Calculates the current index of the event.

        :raises ValueError: if the event is not epoch based or
            if the steps per epoch are too many.
        :return: The current index of the event, which is either the global step
            or the epoch with the fraction of the current step.
        :rtype: float
        """
        if not self.epoch_based:
            return self.global_step
        epoch_full = self.epoch_full
        if epoch_full - self.epoch > 1.0:
            logger.error("Too many steps per epoch for epoch based event")
            raise ValueError("Too many steps per epoch for epoch based event")
        return epoch_full

    @current_index.setter
    def current_index(self, value: float):
        """
        Sets the current index of the event.

        :param value: The current index value.
        :type value: float
        """
        logger.debug("Setting current index: {}", value)
        if not self.epoch_based:
            self.global_step = int(value)
            self.global_batch = (
                self.global_step
                if self.batches_per_step is None or self.batches_per_step < 2
                else self.global_step * self.batches_per_step
            )
        else:
            self.global_step = int(value * self.steps_per_epoch)
            self.global_batch = (
                self.global_step
                if self.batches_per_step is None or self.batches_per_step < 2
                else self.global_step * self.batches_per_step
            )

    def should_update(
        self, start: Optional[float], end: Optional[float], update: Optional[float]
    ) -> bool:
        """
        Determines if the event should trigger an update.

        :param start: The start index to check against, set to None to ignore start.
        :type start: Optional[float]
        :param end: The end index to check against, set to None to ignore end.
        :type end: Optional[float]
        :param update: The update interval, set to None or 0.0 to always update,
            otherwise must be greater than 0.0, defaults to None.
        :type update: Optional[float]
        :return: True if the event should trigger an update, False otherwise.
        :rtype: bool
        """
        current = self.current_index
        logger.debug(
            "Checking if event should update: "
            "current_index={}, start={}, end={}, update={}",
            current,
            start,
            end,
            update,
        )
        if start is not None and current < start:
            return False
        if end is not None and current > end:
            return False
        return update is None or update <= 0.0 or current % update < 1e-10

    def new_instance(self, **kwargs) -> "Event":
        """
        Creates a new instance of the event with the provided keyword arguments.

        :param kwargs: Keyword arguments to set in the new instance.
        :return: A new instance of the event with the provided kwargs.
        :rtype: Event
        """
        logger.debug("Creating new instance of event with kwargs: {}", kwargs)
        instance = deepcopy(self)
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance
