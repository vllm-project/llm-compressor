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
    :param SEQUENTIAL_EPOCH_END: Event type for after a model layer has been calibrated
        for one epoch.
    :param CALIBRATION_EPOCH_END: Event type for after a model has been calibrated
        for one epoch.
    """

    # training and calibration
    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # training
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    BATCH_END = "batch_end"
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"

    # calibration
    SEQUENTIAL_EPOCH_END = "sequential_epoch_end"
    CALIBRATION_EPOCH_END = "calibration_epoch_end"


@dataclass
class Event:
    """
    A class for defining an event that can be triggered during calibration or training

    :param type_: The type of event.
    :type type_: Optional[EventType]
    """

    type_: Optional[EventType] = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
