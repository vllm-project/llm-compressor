from abc import ABC
from typing import List, Optional

import torch

from llmcompressor.core import Event, EventType, State
from llmcompressor.core.event_lifecycle import EventsLifecycle
from llmcompressor.modifiers import Modifier
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)


class EventsMixin(ABC):
    state: State
    modifiers: List[Modifier]

    @EventsLifecycle.initialize
    def initialize(self):
        for modifier in self.modifiers:
            modifier.on_initialize(self.state)
            if modifier.should_start(self.state):
                modifier.on_start(self.state)

        modify_save_pretrained(self.state.model)

    @EventsLifecycle.finalize
    def finalize(self):
        for modifier in self.modifiers:
            if modifier.should_end(self.state):
                modifier.on_end(self.state)
            modifier.on_finalize(self.state)

    @EventsLifecycle.global_step
    def batch_start(self, global_step: Optional[int] = None, **kwargs):
        for modifier in self.modifiers:
            if modifier.should_start(self.state):
                modifier.on_start(self.state)

        event = Event(type_=EventType.BATCH_START, global_step=global_step**kwargs)
        self._handle_event(event)

    def optim_pre_step(self, **kwargs):
        event = Event(type_=EventType.OPTIM_PRE_STEP, **kwargs)
        self._handle_event(event)

    def optim_post_step(self, **kwargs):
        event = Event(type_=EventType.OPTIM_POST_STEP, **kwargs)
        self._handle_event(event)

    def loss_calculated(self, loss: torch.Tensor, **kwargs):
        event = Event(type_=EventType.LOSS_CALCULATED, loss=loss, **kwargs)
        self._handle_event(event)

    def sequential_epoch_end(self, **kwargs):
        event = Event(type_=EventType.SEQUENTIAL_EPOCH_END, **kwargs)
        self._handle_event(event)

    def calibration_epoch_end(self, **kwargs):
        event = Event(type_=EventType.CALIBRATION_EPOCH_END, **kwargs)
        self._handle_event(event)

    def batch_end(self, **kwargs):
        for modifier in self.modifiers:
            if modifier.should_end(self.state):
                modifier.on_end(self.state)

        event = Event(type_=EventType.BATCH_END, **kwargs)
        self._handle_event(event)

    @EventsLifecycle.event
    def _handle_event(self, event: Event):
        for modifier in self.modifiers:
            if modifier.started_ and not modifier.ended_:
                modifier.on_event(self.state, event)
