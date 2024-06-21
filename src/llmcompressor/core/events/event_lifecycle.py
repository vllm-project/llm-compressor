"""
Module for defining and managing the event lifecycle in the LLM Compressor.

This module provides an abstract base class for defining event lifecycles
and methods for retrieving events based on their type,
managing step and batch counts, and triggering events.
"""

from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from llmcompressor.core.events.event import Event, EventType

__all__ = [
    "EventLifecycle",
]


class EventLifecycle(ABC, Event):
    """
    A lifecycle for events to be used in a LLMCompressor session.
    Provides base utilities and defines the contract that
    all inheritors must follow.

    The order in which the events are called is determined by
    the inheritors of this class.

    The expected lifecycle is as follows with optional gradient accumulation:
        for gradient_batches in training loop:
            for batch in gradient_batches or [gradient_batches]:
                BATCH_START
                LOSS_CALCULATED

                if not last batch:
                    BATCH_END
                else:
                    OPTIM_PRE_STEP
                    OPTIM_POST_STEP
                    BATCH_END
    For older flows where the optimizer is wrapped and invocations_per_step > 1:
        for gradient_batches in training loop:
            for invocation in range(invocations_per_step):
                for batch in gradient_batches or [gradient_batches]:
                    BATCH_START
                    LOSS_CALCULATED

                    if not last batch or not last invocation:
                        BATCH_END
                    else:
                        OPTIM_PRE_STEP
                        OPTIM_POST_STEP
                        BATCH_END

    :param type_first: The first event type to be called
    :type type_first: EventType
    :param start: The start event to base the lifecycle off of
    :type start: Event
    """

    def __init__(self, type_first: EventType, start: Event):
        logger.debug(
            "Initializing EventLifecycle with type_first={} and start={}",
            type_first,
            start,
        )
        self.type_first = type_first
        self.last_type = None

        self.steps_per_epoch = start.steps_per_epoch
        self.batches_per_step = start.batches_per_step
        self.invocations_per_step = start.invocations_per_step
        self.global_step = start.global_step
        self.global_batch = start.global_batch

    def events_from_type(self, type_: EventType) -> List[Event]:
        """
        Get the list of events for a given type.

        :param type_: The event type to get the events for
        :type type_: EventType
        :return: The list of events for the given type
        :rtype: List[Event]
        :raises ValueError: If the event type is invalid
        """
        logger.debug("Fetching events from type: {}", type_)
        if type_ == EventType.BATCH_START:
            return self.batch_start_events()
        if type_ == EventType.LOSS_CALCULATED:
            return self.loss_calculated_events()
        if type_ == EventType.OPTIM_PRE_STEP:
            return self.optim_pre_step_events()
        if type_ == EventType.OPTIM_POST_STEP:
            return self.optim_post_step_events()
        if type_ == EventType.BATCH_END:
            return self.batch_end_events()
        logger.error("Invalid event type: {}", type_)
        raise ValueError(f"Invalid event type {type_}")

    def check_batches_per_step_count(self, increment: bool) -> bool:
        """
        Check if the batch count is at the step or step invocation count.
        If batches_per_step is None or < 2, always returns True.

        If invocations_per_step is > 1,
        then returns True for batches matching the invocation.
        Check check_invocations_per_step_count for the invocation count.

        :param increment: Whether to increment the batch count
        :type increment: bool
        :return: True if the batch count is at the step count, False otherwise
        :rtype: bool
        """
        compare_batch = self.global_batch + 1
        at_step = (
            self.batches_per_step is None
            or self.batches_per_step < 2
            or (compare_batch % self.batches_per_step == 0)
        )
        if increment:
            self.global_batch = compare_batch

        return at_step

    def check_invocations_per_step_count(self, increment: bool) -> bool:
        """
        Check if the invocation count is at the step count.
        If invocations_per_step is None or < 2, always returns True.

        :param increment: Whether to increment the step count
        :type increment: bool
        :return: True if the invocation count is at the step count, False otherwise
        :rtype: bool
        """
        compare_step = self.global_step + 1
        at_step = (
            self.invocations_per_step is None
            or self.invocations_per_step < 2
            or (compare_step % self.invocations_per_step == 0)
        )

        if increment:
            self.global_step = compare_step

        return at_step

    @abstractmethod
    def batch_start_events(self) -> List[Event]:
        """Return the list of events to be called for the batch start."""
        raise NotImplementedError()

    @abstractmethod
    def loss_calculated_events(self) -> List[Event]:
        """Return the list of events to be called for the loss calculated."""
        raise NotImplementedError()

    @abstractmethod
    def optim_pre_step_events(self) -> List[Event]:
        """Return the list of events to be called for the optim pre step."""
        raise NotImplementedError()

    @abstractmethod
    def optim_post_step_events(self) -> List[Event]:
        """Return the list of events to be called for the optim post step."""
        raise NotImplementedError()

    @abstractmethod
    def batch_end_events(self) -> List[Event]:
        """Return the list of events to be called for the batch end."""
        raise NotImplementedError()
