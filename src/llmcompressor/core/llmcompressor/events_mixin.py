from abc import ABC
from typing import List

import torch

from llmcompressor.core.llmcompressor.event_lifecycle import EventsLifecycle
from llmcompressor.core import State, EventType, Event

from llmcompressor.modifiers import Modifier
from llmcompressor.transformers.sparsification.compressed_tensors_utils import modify_save_pretrained


class EventsMixin(ABC):
    state: State
    modifiers: List[Modifier]
    
    @EventsLifecycle.validate_initialize
    def initialize(self):
        for modifier in self.modifiers:
            modifier.on_initialize(self.state)

    @EventsLifecycle.validate_finalize
    def finalize(self):
        for modifier in self.modifiers:
            modifier.on_finalize(self.state)
            
        # avoid compressed checkpoints
        # post processing
        modify_save_pretrained(self.state.model)
    
    @EventsLifecycle.handle_global_step
    def batch_start(self, **kwargs):
        # modifiers can only start on batch_start
        for modifier in self.modifiers:
            if modifier.should_start(self.state):
                modifier.on_start(self.state)
        
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

    def batch_end(self, **kwargs):
        # modifiers can only end on batch_end
        for modifier in self.modifiers:
            if modifier.should_end(self.state):
                modifier.on_end(self.state)

        event = Event(type_=EventType.BATCH_END, **kwargs)
        self._handle_event(event)
            
    @EventsLifecycle.validate_event
    def _handle_event(self, event: Event):
        for modifier in self.modifiers:
            modifier.on_event(self.state, event)