# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            calib_data=[],
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
