from typing import List

import pytest

from llmcompressor.core import Event, EventLifecycle, EventType


class DummyEventLifecycle(EventLifecycle):
    def batch_start_events(self) -> List[Event]:
        return [Event(type_=EventType.BATCH_START)]

    def loss_calculated_events(self) -> List[Event]:
        return [Event(type_=EventType.LOSS_CALCULATED)]

    def optim_pre_step_events(self) -> List[Event]:
        return [Event(type_=EventType.OPTIM_PRE_STEP)]

    def optim_post_step_events(self) -> List[Event]:
        return [Event(type_=EventType.OPTIM_POST_STEP)]

    def batch_end_events(self) -> List[Event]:
        return [Event(type_=EventType.BATCH_END)]


@pytest.mark.smoke
def test_event_lifecycle_initialization():
    lifecycle = DummyEventLifecycle(
        type_first=EventType.BATCH_START,
        start=Event(
            steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
        ),
    )

    assert lifecycle.type_first == EventType.BATCH_START
    assert lifecycle.last_type is None
    assert lifecycle.steps_per_epoch == 10
    assert lifecycle.batches_per_step == 2
    assert lifecycle.invocations_per_step == 1
    assert lifecycle.global_step == 0
    assert lifecycle.global_batch == 0


@pytest.mark.smoke
def test_check_step_batches_count():
    lifecycle = DummyEventLifecycle(
        type_first=EventType.BATCH_START,
        start=Event(
            steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
        ),
    )

    assert lifecycle.check_batches_per_step_count(increment=True) is False
    assert lifecycle.global_batch == 1

    assert lifecycle.check_batches_per_step_count(increment=False) is True
    assert lifecycle.global_batch == 1

    assert lifecycle.check_batches_per_step_count(increment=True) is True
    assert lifecycle.global_batch == 2


@pytest.mark.smoke
def test_check_default_step_invocations_count():
    lifecycle = DummyEventLifecycle(
        type_first=EventType.BATCH_START,
        start=Event(
            steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
        ),
    )

    assert lifecycle.check_invocations_per_step_count(increment=True) is True
    assert lifecycle.global_step == 1

    assert lifecycle.check_invocations_per_step_count(increment=False) is True
    assert lifecycle.global_step == 1

    assert lifecycle.check_invocations_per_step_count(increment=True) is True
    assert lifecycle.global_step == 2


@pytest.mark.smoke
def test_events_from_type():
    lifecycle = DummyEventLifecycle(
        type_first=EventType.BATCH_START,
        start=Event(
            steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
        ),
    )

    events = lifecycle.events_from_type(EventType.BATCH_START)
    assert len(events) == 1
    assert events[0].type_ == EventType.BATCH_START

    events = lifecycle.events_from_type(EventType.LOSS_CALCULATED)
    assert len(events) == 1
    assert events[0].type_ == EventType.LOSS_CALCULATED

    events = lifecycle.events_from_type(EventType.OPTIM_PRE_STEP)
    assert len(events) == 1
    assert events[0].type_ == EventType.OPTIM_PRE_STEP

    events = lifecycle.events_from_type(EventType.OPTIM_POST_STEP)
    assert len(events) == 1
    assert events[0].type_ == EventType.OPTIM_POST_STEP

    events = lifecycle.events_from_type(EventType.BATCH_END)
    assert len(events) == 1
    assert events[0].type_ == EventType.BATCH_END


@pytest.mark.regression
def test_check_step_invocations_count():
    lifecycle = DummyEventLifecycle(
        type_first=EventType.BATCH_START,
        start=Event(
            steps_per_epoch=10,
            batches_per_step=2,
            global_step=0,
            global_batch=0,
            invocations_per_step=2,
        ),
    )

    assert lifecycle.check_invocations_per_step_count(increment=True) is False
    assert lifecycle.global_step == 1

    assert lifecycle.check_invocations_per_step_count(increment=False) is True
    assert lifecycle.global_step == 1

    assert lifecycle.check_invocations_per_step_count(increment=True) is True
    assert lifecycle.global_step == 2


@pytest.mark.regression
def test_not_implemented_errors():
    with pytest.raises(TypeError):

        class TestIncompleteEventLifecycle(EventLifecycle):
            pass

        TestIncompleteEventLifecycle(None, None)

    class IncompleteEventLifecycle(EventLifecycle):
        def batch_start_events(self) -> List[Event]:
            return super().batch_start_events()

        def loss_calculated_events(self) -> List[Event]:
            return super().loss_calculated_events()

        def optim_pre_step_events(self) -> List[Event]:
            return super().optim_pre_step_events()

        def optim_post_step_events(self) -> List[Event]:
            return super().optim_post_step_events()

        def batch_end_events(self) -> List[Event]:
            return super().batch_end_events()

    start_event = Event(
        steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
    )
    lifecycle = IncompleteEventLifecycle(
        type_first=EventType.BATCH_START, start=start_event
    )

    with pytest.raises(NotImplementedError):
        lifecycle.batch_start_events()

    with pytest.raises(NotImplementedError):
        lifecycle.loss_calculated_events()

    with pytest.raises(NotImplementedError):
        lifecycle.optim_pre_step_events()

    with pytest.raises(NotImplementedError):
        lifecycle.optim_post_step_events()

    with pytest.raises(NotImplementedError):
        lifecycle.batch_end_events()
