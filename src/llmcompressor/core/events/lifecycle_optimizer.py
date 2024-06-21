"""
Module for defining and managing optimizer event lifecycles in the LLM Compressor.

This module provides a class for defining event lifecycles when the optimizer is wrapped
to invoke the event lifecycle and no callbacks are used.
"""

from typing import List

from loguru import logger

from llmcompressor.core.events.event import Event, EventType
from llmcompressor.core.events.event_lifecycle import EventLifecycle

__all__ = [
    "OptimizerEventLifecycle",
]


class OptimizerEventLifecycle(EventLifecycle):
    """
    An event lifecycle for when the optimizer is wrapped to invoke the event lifecycle
    and no callbacks are used.

    For all flows with the OptimizerEventLifecycle, the optimizer is wrapped
    to trigger around the step function on the optimizer.
    loss_calculated is optional, but if used it must be consistently
    called before optimizer steps.

    The expected lifecycle is as follows with no gradient accumulation:
        for batch in training loop:
            if loss callbacks:
                loss_calculated() -> [BATCH_START, LOSS_CALCULATED]
                optim.step() -> [BATCH_START, OPTIM_PRE_STEP,
                                 OPTIM_POST_STEP, BATCH_END]
            else:
                optim.step() -> [BATCH_START, OPTIM_PRE_STEP,
                                 OPTIM_POST_STEP, BATCH_END]
    For gradient accumulation:
        for gradient_batches in training loop:
            for batch in gradient_batches:
                if loss callbacks:
                    if not last batch:
                        loss_calculated() -> [BATCH_START, LOSS_CALCULATED, BATCH_END]
                    else:
                        loss_calculated() -> [BATCH_START, LOSS_CALCULATED]
                        optim.step() -> [OPTIM_PRE_STEP, OPTIM_POST_STEP, BATCH_END]
                else:
                    if last batch:
                        optim.step() -> [BATCH_START, OPTIM_PRE_STEP,
                                         OPTIM_POST_STEP, BATCH_END]
    For older amp scale flows that use invocations_per_step > 1:
        for batch in training loop:
            for invocation in range(invocations_per_step):
                if not last invocation:
                    if loss callbacks:
                        loss_calculated() -> [BATCH_START, LOSS_CALCULATED]
                        optim.step() -> [BATCH_END]
                    else:
                        optim.step() -> [BATCH_START, BATCH_END]
                else:
                    if loss callbacks:
                        loss_calculated() -> [BATCH_START, LOSS_CALCULATED]
                        optim.step() -> [OPTIM_PRE_STEP, OPTIM_POST_STEP, BATCH_END]
                    else:
                        optim.step() -> [BATCH_START, OPTIM_PRE_STEP,
                                         OPTIM_POST_STEP, BATCH_END]

        - batch_start: must not be invoked, auto triggered
            from loss calculated if that is called, otherwise from pre_step
        - loss_calculated: optional pathway and invoked through callbacks.
            It must be called as the first event if used,
            and after optim post step or batch end for subsequent calls.
        - batch_end: must not be invoked, auto triggered from optim_post_step
        - optim_pre_step: must be called before optim_post_step
        - optim_post_step: must be called only once after optim_pre_step
    """

    def __init__(self, type_first: EventType, start: Event):
        """
        Initialize the OptimizerEventLifecycle with the first type and start event.

        :param type_first: The first event type to be called
        :type type_first: EventType
        :param start: The start event to base the lifecycle off of
        :type start: Event
        """
        super().__init__(type_first=type_first, start=start)
        self.skip_post_step = False

    def batch_start_events(self) -> List[Event]:
        """
        Raises a ValueError as this method should not be called.

        :raises ValueError: If invoked as this should not be called
        """
        logger.error("batch start should not be invoked when only wrapped optim")
        raise ValueError("batch start should not be invoked when only wrapped optim")

    def loss_calculated_events(self) -> List[Event]:
        """
        Return the list of events to be called for the loss calculated.

        :return: The list of events to be called for the loss calculated
        :rtype: List[Event]
        :raises ValueError: If invoked before loss calculation
        """
        if self.type_first != EventType.LOSS_CALCULATED:
            logger.error("loss calculated must be called first for wrapped optim")
            raise ValueError("loss calculated must be called first for wrapped optim")

        if self.last_type not in {
            EventType.LOSS_CALCULATED,
            EventType.OPTIM_POST_STEP,
            None,
        }:
            logger.error(
                "loss calculated must be called after batch end or optim post step"
            )
            raise ValueError(
                "loss calculated must be called after batch end or optim post step"
            )

        self.last_type = EventType.LOSS_CALCULATED

        if not self.check_batches_per_step_count(increment=True):
            logger.debug(
                "Loss calculated event processed, "
                "step not ready at global_step: {} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            return [
                self.new_instance(type_=EventType.BATCH_START),
                self.new_instance(type_=EventType.LOSS_CALCULATED),
                self.new_instance(type_=EventType.BATCH_END),
            ]
        else:
            logger.debug(
                "Loss calculated event processed, "
                "step ready at global_step: {} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            return [
                self.new_instance(type_=EventType.BATCH_START),
                self.new_instance(type_=EventType.LOSS_CALCULATED),
            ]

    def optim_pre_step_events(self) -> List[Event]:
        """
        Return the list of events to be called for the optim pre step.

        :return: The list of events to be called for the optim pre step
        :rtype: List[Event]
        :raises ValueError: If optim pre step is not called before optim post step
            or after loss calculated
        """
        if self.type_first == EventType.LOSS_CALCULATED:
            # handle loss calculated case where gradient accumulation
            # is automatically handled by the loss callbacks

            if self.last_type != EventType.LOSS_CALCULATED:
                logger.error("optim pre step must be called after loss calculated")
                raise ValueError("optim pre step must be called after loss calculated")

            self.last_type = EventType.OPTIM_PRE_STEP

            if not self.check_invocations_per_step_count(increment=False):
                logger.debug(
                    "Optim pre step event processed, "
                    "but invocations not ready at global_step: {} and global_batch: {}",
                    self.global_step,
                    self.global_batch,
                )
                self.skip_post_step = True
                return []
            else:
                logger.debug(
                    "Optim pre step event processed, "
                    "invocations ready at global_step: {} and global_batch: {}",
                    self.global_step,
                    self.global_batch,
                )
                self.skip_post_step = False
                return [self.new_instance(type_=EventType.OPTIM_PRE_STEP)]

        # handle no callbacks case to emulate batch events for gradient accumulation
        if self.last_type not in {EventType.OPTIM_POST_STEP, None}:
            logger.error(
                "optim pre step must be called at the start or after optim post step"
            )
            raise ValueError(
                "optim pre step must be called at the start or after optim post step"
            )

        batch_events = [
            self.new_instance(type_=EventType.BATCH_START),
        ]
        while not self.check_batches_per_step_count(increment=True):
            batch_events.append(self.new_instance(type_=EventType.BATCH_END))
            batch_events.append(self.new_instance(type_=EventType.BATCH_START))

        self.last_type = EventType.OPTIM_PRE_STEP

        if not self.check_invocations_per_step_count(increment=False):
            logger.debug(
                "Optim pre step event processed, "
                "but invocations not ready at global_step: {} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            self.skip_post_step = True
            return batch_events
        else:
            logger.debug(
                "Optim pre step event processed, "
                "invocations ready at global_step: {} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            self.skip_post_step = False
            return batch_events + [self.new_instance(type_=EventType.OPTIM_PRE_STEP)]

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

        if self.skip_post_step:
            logger.debug(
                "Skipping optim post step event at global_step: "
                "{} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            return [self.new_instance(type_=EventType.BATCH_END)]
        else:
            logger.debug(
                "Optim post step event processed at global_step: "
                "{} and global_batch: {}",
                self.global_step,
                self.global_batch,
            )
            return [
                self.new_instance(type_=EventType.OPTIM_POST_STEP),
                self.new_instance(type_=EventType.BATCH_END),
            ]

    def batch_end_events(self) -> List[Event]:
        """
        Raises a ValueError as this method should not be called.

        :raises ValueError: If invoked as this should not be called
        """
        logger.error("batch end should not be invoked when only wrapped optim")
        raise ValueError("batch end should not be invoked when only wrapped optim")
