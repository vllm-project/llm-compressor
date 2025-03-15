from abc import ABC

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

    # step lifecycle
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"


class Event:
    _type: EventType = None

    def __init__(self, type: EventType, **kwargs):
        # validate kwargs
        if type is EventType.LOSS_CALCULATED:
            if "loss" not in kwargs:
                raise TypeError(f"Expected argument `loss` for event of type {type}")
            
        for key, value in kwargs:
            setattr(self, key, value)
        
    @property
    def type(self):
        return self._type
