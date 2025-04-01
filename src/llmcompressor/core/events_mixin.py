from abc import ABC
from typing import TYPE_CHECKING, List, Optional

import torch

from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)

from .event import Event, EventType
from .event_lifecycle import EventsLifecycle
from .state import State

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier


class EventsMixin(ABC):
    state: State
    modifiers: List["Modifier"]

    @EventsLifecycle.initialize
    def initialize(self, **kwargs):
        for modifier in self.modifiers:
            modifier.on_initialize(self.state, **kwargs)
            if modifier.should_start(self.state):
                modifier.on_start(self.state)

        modify_save_pretrained(self.state.model)

    @EventsLifecycle.finalize
    def finalize(self, **kwargs):
        for modifier in self.modifiers:
            if not modifier.ended_:
                modifier.on_end(self.state)
            modifier.on_finalize(self.state, **kwargs)

    @EventsLifecycle.global_step
    def batch_start(self, global_step: Optional[int] = None, **kwargs):
        for modifier in self.modifiers:
            if modifier.should_start(self.state):
                modifier.on_start(self.state)

        event = Event(type_=EventType.BATCH_START, global_step=global_step, **kwargs)
        self._handle_event(event)

    def batch_end(self, **kwargs):
        for modifier in self.modifiers:
            if modifier.should_end(self.state):
                modifier.on_end(self.state)

        event = Event(type_=EventType.BATCH_END, **kwargs)
        self._handle_event(event)

    def optim_pre_step(self, **kwargs):
        event = Event(type_=EventType.OPTIM_PRE_STEP, **kwargs)
        self._handle_event(event)

    def optim_post_step(self, **kwargs):
        event = Event(type_=EventType.OPTIM_POST_STEP, **kwargs)
        self._handle_event(event)

    def loss_calculated(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        self.state.loss = loss  # may be modified by modifiers
        event = Event(type_=EventType.LOSS_CALCULATED, loss=loss, **kwargs)
        self._handle_event(event)
        return self.state.loss

    def sequential_epoch_end(self, **kwargs):
        event = Event(type_=EventType.SEQUENTIAL_EPOCH_END, **kwargs)
        self._handle_event(event)

    def calibration_epoch_end(self, **kwargs):
        event = Event(type_=EventType.CALIBRATION_EPOCH_END, **kwargs)
        self._handle_event(event)

    @EventsLifecycle.event
    def _handle_event(self, event: Event):
        for modifier in self.modifiers:
            if modifier.started_ and not modifier.ended_:
                modifier.on_event(self.state, event)
