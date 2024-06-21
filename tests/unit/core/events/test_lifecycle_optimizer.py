import pytest

from llmcompressor.core.events.event import Event, EventType
from llmcompressor.core.events.lifecycle_optimizer import OptimizerEventLifecycle


@pytest.mark.smoke
def test_initialization():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    assert lifecycle.type_first == EventType.LOSS_CALCULATED
    assert lifecycle.steps_per_epoch == 10
    assert lifecycle.batches_per_step == 1
    assert lifecycle.global_step == 0
    assert lifecycle.global_batch == 0


@pytest.mark.smoke
def test_lifecycle():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.OPTIM_PRE_STEP, start=start_event
    )

    events = lifecycle.optim_pre_step_events()
    assert events[0].type_ == EventType.BATCH_START
    assert events[1].type_ == EventType.OPTIM_PRE_STEP

    events = lifecycle.optim_post_step_events()
    assert events[0].type_ == EventType.OPTIM_POST_STEP
    assert events[1].type_ == EventType.BATCH_END


@pytest.mark.smoke
def test_lifecycle_with_loss():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    events = lifecycle.loss_calculated_events()
    assert events[0].type_ == EventType.BATCH_START
    assert events[1].type_ == EventType.LOSS_CALCULATED

    events = lifecycle.optim_pre_step_events()
    assert events[0].type_ == EventType.OPTIM_PRE_STEP

    events = lifecycle.optim_post_step_events()
    assert events[0].type_ == EventType.OPTIM_POST_STEP
    assert events[1].type_ == EventType.BATCH_END


@pytest.mark.sanity
def test_loss_calculated_events():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    events = lifecycle.loss_calculated_events()
    assert len(events) == 2
    assert events[0].type_ == EventType.BATCH_START
    assert events[1].type_ == EventType.LOSS_CALCULATED


@pytest.mark.sanity
def test_optim_pre_step_events():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    lifecycle.loss_calculated_events()
    events = lifecycle.optim_pre_step_events()
    assert len(events) == 1
    assert events[0].type_ == EventType.OPTIM_PRE_STEP

    with pytest.raises(ValueError):
        lifecycle.optim_pre_step_events()


@pytest.mark.sanity
def test_optim_post_step_events():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    lifecycle.loss_calculated_events()
    lifecycle.optim_pre_step_events()
    events = lifecycle.optim_post_step_events()
    assert len(events) == 2
    assert events[0].type_ == EventType.OPTIM_POST_STEP
    assert events[1].type_ == EventType.BATCH_END

    with pytest.raises(ValueError):
        lifecycle.optim_post_step_events()


@pytest.mark.regression
def test_lifecycle_gradient_accumulation():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.OPTIM_PRE_STEP, start=start_event
    )

    events = lifecycle.optim_pre_step_events()
    assert events[0].type_ == EventType.BATCH_START
    assert events[1].type_ == EventType.BATCH_END
    assert events[2].type_ == EventType.BATCH_START
    assert events[3].type_ == EventType.OPTIM_PRE_STEP

    events = lifecycle.optim_post_step_events()
    assert events[0].type_ == EventType.OPTIM_POST_STEP
    assert events[1].type_ == EventType.BATCH_END


@pytest.mark.regression
def test_lifecycle_gradient_accumulation_loss_calculated():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=2, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    for index in range(2):
        events = lifecycle.loss_calculated_events()

        assert events[0].type_ == EventType.BATCH_START
        assert events[1].type_ == EventType.LOSS_CALCULATED

        if index == 0:
            assert events[2].type_ == EventType.BATCH_END

    events = lifecycle.optim_pre_step_events()
    assert events[0].type_ == EventType.OPTIM_PRE_STEP

    events = lifecycle.optim_post_step_events()
    assert events[0].type_ == EventType.OPTIM_POST_STEP
    assert events[1].type_ == EventType.BATCH_END


@pytest.mark.regression
def test_invalid_event_order():
    start_event = Event(
        steps_per_epoch=10, batches_per_step=1, global_step=0, global_batch=0
    )
    lifecycle = OptimizerEventLifecycle(
        type_first=EventType.LOSS_CALCULATED, start=start_event
    )

    with pytest.raises(ValueError):
        lifecycle.optim_pre_step_events()

    events = lifecycle.loss_calculated_events()
    assert events

    with pytest.raises(ValueError):
        lifecycle.optim_post_step_events()

    events = lifecycle.optim_pre_step_events()
    assert events

    with pytest.raises(ValueError):
        lifecycle.batch_end_events()

    events = lifecycle.optim_post_step_events()
    assert events
