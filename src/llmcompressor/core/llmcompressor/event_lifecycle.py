from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from loguru import logger

from llmcompressor.core.events import Event, EventType
from llmcompressor.utils.singleton import SingletonMixin

if TYPE_CHECKING:
    from llmcompressor.core.llmcompressor.events_mixin import EventsMixin


class EventsLifecycle(SingletonMixin):
    auto_step: Optional[bool] = None
    event_order: List[EventType] = [
        EventType.BATCH_START,
        EventType.LOSS_CALCULATED,
        EventType.OPTIM_PRE_STEP,
        EventType.OPTIM_POST_STEP,
        EventType.BATCH_END,
    ]
    last_event_type: Optional[EventType] = EventType.BATCH_END
    initialized: bool = False
    finalized: bool = False

    @classmethod
    def initalize(cls, fn: Callable[[Any], Any]):
        def validator(self: "EventsMixin", **kwargs):
            if cls.initialized:
                raise ValueError("Cannot initialize twice")
            cls.initialized = True
            cls.finalized = False

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def finalize(cls, fn: Callable[[Any], Any]):
        def validator(self: "EventsMixin", **kwargs):
            if not cls.initialized:
                raise ValueError("Cannot finalize before initializing")
            if cls.finalized:
                raise ValueError("Cannot finalize twice")
            cls.finalized = True
            cls.initialized = False

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def global_step(cls, fn: Callable[[Any], Any]):
        def validator(self: "EventsMixin", global_step: Optional[int] = None, **kwargs):
            # configure auto step
            if cls.auto_step is None:
                if global_step is None:
                    logger.info(
                        "No global_step was passed to batch_start event, "
                        "auto-stepping based on batches"
                    )
                    cls.auto_step = True
                else:
                    cls.auto_step = False

            # auto step
            if global_step is None:
                if not cls.auto_step:
                    raise ValueError(
                        "Cannot auto-step batches if global_step was "
                        "previously passed to batch_start event"
                    )
                global_step = self.state.current_index + 1
            else:
                if cls.auto_step:
                    raise ValueError(
                        "Cannot auto-step batches if global_step "
                        "was passed to batch_start event"
                    )

            # validate order
            if global_step <= self.state.current_index:
                raise ValueError("global_step must be greater than the current index")

            self.state.current_index = global_step

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def event(cls, fn: Callable[[Any], Any]):
        def validator(self: "EventsMixin", event: Event):
            event_type = event.type

            # ignore unhandled events
            if event_type not in cls.event_order:
                return

            # validate
            if event_type == EventType.BATCH_START:
                valid = cls.last_event_type != EventType.BATCH_START
            else:
                last_event_index = cls.event_order.index(cls.last_event_type)
                curr_event_index = cls.event_order.index(event_type)
                valid = last_event_index <= curr_event_index

            if not valid:
                raise ValueError(
                    f"Lifecycle events must appear in order: {cls.event_order}. "
                    f"Instead, {cls.last_event_type} was called before {event_type}"
                )

            cls.last_event_type = event_type

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def _wrap_with_validation(
        cls, fn: Callable[[Any], Any], validator: Callable[[Any], Any]
    ) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            validator(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapped
