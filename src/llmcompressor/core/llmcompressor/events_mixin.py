from abc import ABC
from typing import List

import torch

from llmcompressor.core import Event, EventType, State
from llmcompressor.core.llmcompressor.event_lifecycle import EventsLifecycle
from llmcompressor.modifiers import Modifier
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)

# (1) can remove event arg after the default pathway is either removed or
# no longer depends on event to get current_index


class EventsMixin(ABC):
    state: State
    modifiers: List[Modifier]

    @EventsLifecycle.initialize
    def initialize(self):
        for modifier in self.modifiers:
            modifier.on_initialize(self.state)
            if modifier.lc_should_start(self.state):
                modifier.on_start(self.state, None)  # (1)

    @EventsLifecycle.finalize
    def finalize(self):
        for modifier in self.modifiers:
            modifier.on_finalize(self.state)

        # TODO: log info stating that save_pretrained has been modified
        # TODO: make sure wrapped function can access new recipe and processor
        modify_save_pretrained(self.state.model)

    def update_state(self, **kwargs):
        self.state.update(**kwargs)
        # if future modifiers require update, do that update here

    @EventsLifecycle.global_step
    def batch_start(self, **kwargs):
        for modifier in self.modifiers:
            if modifier.lc_should_start(self.state):
                modifier.on_start(self.state, None)  # (1)

        event = Event(type_=EventType.BATCH_START, **kwargs)
        self._handle_event(event)

    def pre_optim(self, **kwargs):
        event = Event(type_=EventType.OPTIM_PRE_STEP, **kwargs)
        self._handle_event(event)

    def post_optim(self, **kwargs):
        event = Event(type_=EventType.OPTIM_POST_STEP, **kwargs)
        self._handle_event(event)

    def update_loss(self, loss: torch.Tensor, **kwargs):
        event = Event(type_=EventType.LOSS_CALCULATED, loss=loss, **kwargs)
        self._handle_event(event)

    def sequential_epoch_end(self, **kwargs):
        event = Event(type_=EventType.SEQUENTIAL_EPOCH_END, **kwargs)
        self._handle_event(event)

    def calibration_epoch_end(self, **kwargs):
        print("calibration_epoch_end")
        event = Event(type_=EventType.CALIBRATION_EPOCH_END, **kwargs)
        self._handle_event(event)

    def batch_end(self, **kwargs):
        for modifier in self.modifiers:
            if modifier.lc_should_end(self.state):
                modifier.on_end(self.state, None)  # (1)

        event = Event(type_=EventType.BATCH_END, **kwargs)
        self._handle_event(event)

    @EventsLifecycle.event
    def _handle_event(self, event: Event):
        for modifier in self.modifiers:
            if modifier.started_ and not modifier.ended_:
                modifier.on_event(self.state, event)
