"""
Module for defining and managing events in the LLM Compressor.

This module provides an Enum for different event types and a class for creating and
managing events, including methods for calculating event properties and triggering
updates based on specified intervals.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

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

    :param INITIALIZE: Event type for initialization.
    :param FINALIZE: Event type for finalization.
    :param BATCH_START: Event type for the start of a batch.
    :param LOSS_CALCULATED: Event type for when loss is calculated.
    :param BATCH_END: Event type for the end of a batch.
    :param OPTIM_PRE_STEP: Event type for pre-optimization step.
    :param OPTIM_POST_STEP: Event type for post-optimization step.
    """

    # training lifecycle
    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # batch lifecycle
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    BATCH_END = "batch_end"
    SEQUENTIAL_EPOCH_END = "sequential_epoch_end"
    CALIBRATION_EPOCH_END = "calibration_epoch_end"

    # step lifecycle
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"


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

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
