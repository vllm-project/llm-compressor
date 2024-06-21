"""
Module for defining and managing callback event lifecycles in the LLM Compressor.

This module provides a class for defining event lifecycles when callbacks
are used to communicate the state of the training pipeline.
"""

from typing import List

from loguru import logger

from llmcompressor.core.events.event import Event, EventType
from llmcompressor.core.events.event_lifecycle import EventLifecycle

__all__ = [
    "CallbacksEventLifecycle",
]


class CallbacksEventLifecycle(EventLifecycle):
    """
    An event lifecycle for when callbacks are used to communicate the
    state of the training pipeline.

    The expected lifecycle is as follows with optional gradient accumulation:
        for gradient_batches in training loop:
            for batch in gradient_batches or [gradient_batches]:
                batch_start() -> [BATCH_START]
                loss_calculated() -> [LOSS_CALCULATED]

                if not last batch:
                    batch_end -> [BATCH_END]
                else:
                    optim_pre_step() -> [OPTIM_PRE_STEP]
                    optim_post_step() -> [OPTIM_POST_STEP]
                    batch_end -> [BATCH_END]

    Which gives the following logic:
        - BATCH_START: must be called first or after OPTIM_POST_STEP
        - LOSS_CALCULATED: must be called after BATCH_START and
            before BATCH_END or OPTIM_POST_STEP
        - OPTIM_PRE_STEP: must be called after LOSS_CALCULATED
            and before OPTIM_POST_STEP
        - OPTIM_POST_STEP: must be called after OPTIM_PRE_STEP and before BATCH_END
        - BATCH_END: must be called after LOSS_CALCULATED or OPTIM_POST_STEP
    """

    def __init__(self, type_first: EventType, start: Event):
        """
        Initialize the CallbacksEventLifecycle.

        :param type_first: The first event type to be called
        :type type_first: EventType
        :param start: The start event to base the lifecycle off of
        :type start: Event
        """
        super().__init__(type_first=type_first, start=start)
        self.skip_post_step = False

    def batch_start_events(self) -> List[Event]:
        """
        Return the list of events to be called for the batch start.

        :return: The list of events to be called for the batch start
        :rtype: List[Event]
        :raises ValueError: If batch start is not called first or if it
            is not called after batch end
        """
        if self.type_first != EventType.BATCH_START:
            logger.error("batch start must be called first for callbacks")
            raise ValueError("batch start must be called first for callbacks")

        if self.last_type not in {None, EventType.BATCH_END, EventType.OPTIM_POST_STEP}:
            logger.error("batch start must be called after batch end")
            raise ValueError("batch start must be called after batch end")

        self.last_type = EventType.BATCH_START
        step_ready = self.check_batches_per_step_count(increment=True)
        logger.debug(
            "Batch start event processed with step_ready={}, "
            "global_step={}, and global_batch={}",
            step_ready,
            self.global_step,
            self.global_batch,
        )

        return [self.new_instance(type_=EventType.BATCH_START)]

    def loss_calculated_events(self) -> List[Event]:
        """
        Return the list of events to be called for the loss calculated.

        :return: The list of events to be called for the loss calculated
        :rtype: List[Event]
        :raises ValueError: If loss calculated is not called after batch start
        """
        if self.last_type != EventType.BATCH_START:
            logger.error("loss calculated must be called after batch start")
            raise ValueError("loss calculated must be called after batch start")

        self.last_type = EventType.LOSS_CALCULATED
        logger.debug(
            "Loss calculated event processed with global_batch={} and global_step={}",
            self.global_batch,
            self.global_step,
        )

        return [self.new_instance(type_=EventType.LOSS_CALCULATED)]

    def optim_pre_step_events(self) -> List[Event]:
        """
        Return the list of events to be called for the optim pre step.

        :return: The list of events to be called for the optim pre step
        :rtype: List[Event]
        :raises ValueError: If optim pre step is not called after batch start
            or loss calculated
        """
        if self.last_type not in {EventType.LOSS_CALCULATED}:
            logger.error("optim pre step must be called after loss calculated")
            raise ValueError("optim pre step must be called after loss calculated")

        self.last_type = EventType.OPTIM_PRE_STEP
        at_invocation = self.check_invocations_per_step_count(increment=True)
        logger.debug(
            "Optim pre step event processed with at_invocation={}, "
            "global_step={}, and global_batch={}",
            at_invocation,
            self.global_step,
            self.global_batch,
        )

        if not at_invocation:
            self.skip_post_step = True
            return []
        else:
            self.skip_post_step = False
            return [self.new_instance(type_=EventType.OPTIM_PRE_STEP)]

    def optim_post_step_events(self) -> List[Event]:
        """
        Return the list of events to be called for the optim post step.

        :return: The list of events to be called for the optim post step
        :rtype: List[Event]
        :raises ValueError: If optim post step is not called after optim pre step
        """
        if self.last_type != EventType.OPTIM_PRE_STEP:
            logger.error("optim post step must be called after optim pre step")
            raise ValueError("optim post step must be called after optim pre step")

        self.last_type = EventType.OPTIM_POST_STEP
        logger.debug(
            "Optim post step event processed with global_batch={} and global_step={}",
            self.global_batch,
            self.global_step,
        )

        if self.skip_post_step:
            return []
        else:
            return [self.new_instance(type_=EventType.OPTIM_POST_STEP)]

    def batch_end_events(self) -> List[Event]:
        """
        Return the list of events to be called for the batch end.

        :return: The list of events to be called for the batch end
        :rtype: List[Event]
        :raises ValueError: If batch end is not called after optim post step,
            loss calculated, or batch start
        """
        if self.last_type not in {
            EventType.OPTIM_POST_STEP,
            EventType.LOSS_CALCULATED,
        }:
            logger.error(
                "batch end must be called after loss calculated or optim post step"
            )
            raise ValueError(
                "batch end must be called after loss calculated or optim post step"
            )

        self.last_type = EventType.BATCH_END
        logger.debug(
            "Batch end event processed with global_batch={} and global_step={}",
            self.global_batch,
            self.global_step,
        )

        return [self.new_instance(type_=EventType.BATCH_END)]
