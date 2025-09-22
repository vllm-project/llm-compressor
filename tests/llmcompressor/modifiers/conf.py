from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from llmcompressor.core import Event, EventType, State


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

    def update_modifier(self, modifier, event_type):
        event = Event(event_type=event_type)
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
