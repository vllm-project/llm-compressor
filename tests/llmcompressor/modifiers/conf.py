from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from llmcompressor.core import State
from llmcompressor.core.events import EventType
from llmcompressor.core.lifecycle import CallbacksEventLifecycle
from llmcompressor.modifiers.factory import ModifierFactory


def setup_modifier_factory():
    ModifierFactory.refresh()
    assert ModifierFactory._loaded, "ModifierFactory not loaded"


class LifecyleTestingHarness:
    def __init__(
        self,
        model=None,
        optimizer=None,
        device="cpu",
        start=0,
    ):
        self.state = State()
        self.state.update(
            model=model,
            device=device,
            optimizer=optimizer,
            start=start,
            steps_per_epoch=1,
            calib_data=DataLoader(MagicMock(__len__=lambda _: 0, column_names=[])),
        )

        self.event_lifecycle = CallbacksEventLifecycle(
            type_first=EventType.BATCH_START, start=self.state.start_event
        )

    def update_modifier(self, modifier, event_type):
        events = self.event_lifecycle.events_from_type(event_type)
        for event in events:
            modifier.update_event(self.state, event=event)

    def get_state(self):
        return self.state

    def trigger_modifier_for_epochs(self, modifier, num_epochs):
        for _ in range(num_epochs):
            self.update_modifier(modifier, EventType.BATCH_START)
            self.update_modifier(modifier, EventType.LOSS_CALCULATED)
            self.update_modifier(modifier, EventType.OPTIM_PRE_STEP)
            self.update_modifier(modifier, EventType.OPTIM_POST_STEP)
            self.update_modifier(modifier, EventType.BATCH_END)
