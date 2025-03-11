import pytest

from llmcompressor.core import Event, EventType


@pytest.mark.smoke
def test_event_epoch_based():
    event = Event(steps_per_epoch=10)
    assert event.epoch_based is True


@pytest.mark.smoke
def test_event_epoch():
    event = Event(steps_per_epoch=10, global_step=25)
    assert event.epoch == 2


@pytest.mark.smoke
def test_event_epoch_full():
    event = Event(steps_per_epoch=10, global_step=25)
    assert event.epoch_full == 2.5


@pytest.mark.smoke
def test_event_epoch_step():
    event = Event(steps_per_epoch=10, global_step=25)
    assert event.epoch_step == 5


@pytest.mark.smoke
def test_event_epoch_batch():
    event = Event(
        steps_per_epoch=10, global_step=25, batches_per_step=2, global_batch=50
    )
    assert event.epoch_batch == 10


@pytest.mark.smoke
def test_event_current_index():
    event = Event(steps_per_epoch=10, global_step=25)
    assert event.current_index == 2.5


@pytest.mark.smoke
def test_event_should_update():
    event = Event(steps_per_epoch=10, global_step=25)
    assert event.should_update(start=0, end=30, update=2.5) is True
    assert event.should_update(start=0, end=20, update=5) is False
    assert event.should_update(start=0, end=30, update=0) is True


@pytest.mark.smoke
def test_event_new_instance():
    event = Event(type_=EventType.INITIALIZE, global_step=25)
    new_event = event.new_instance(global_step=30)
    assert new_event.global_step == 30
    assert new_event.type_ == EventType.INITIALIZE
