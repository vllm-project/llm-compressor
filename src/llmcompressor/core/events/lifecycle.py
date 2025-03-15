from typing import List, Callable, Any, Optional, TYPE_CHECKING

from loguru import logger

from llmcompressor.core.events import Event, EventType

if TYPE_CHECKING:
    from llmcompressor.core.events.events_mixin import EventsMixin


class EventsLifecycle:
    auto_step: Optional[bool] = None
    event_order: List[EventType] =[
        EventType.BATCH_START,
        EventType.LOSS_CALCULATED,
        EventType.OPTIM_PRE_STEP,
        EventType.OPTIM_POST_STEP,
        EventType.BATCH_END,
    ]
    last_event_type: Optional[EventType] = EventType.BATCH_END

    @classmethod
    def validate_initialize(cls, fn: Callable[[Any], Any]):
        return fn
        
    @classmethod
    def validate_finalize(cls, fn: Callable[[Any], Any]):
        return fn
    
    @classmethod
    def handle_global_step(cls, fn: Callable[[Any], Any]):
        def wrapped(self: "EventsMixin", global_step: Optional[int] = None, **kwargs):
            # configure auto step based on first 
            if cls.auto_step is None:
                if global_step is None:
                    logger.info("No global_step was passed to batch_start event, auto-stepping based on batches")
                    cls.auto_step = True
                else:
                    cls.auto_step = None

            # auto step
            if global_step is None:
                if not cls.auto_step:
                    raise ValueError("Cannot auto-step batches if global_step was previously passed to batch_start event")
                
                global_step = self.state.current_index + 1

            else:
                if cls.auto_step:
                    raise ValueError("Cannot auto-step batches if global_step was passed to batch_start event")

            # validate ordering
            if global_step <= self.state.current_index:
                raise ValueError("global_step")
                
            self.state.current_index = global_step
            return fn(self, global_step=global_step, **kwargs)

        return wrapped

    @classmethod
    def validate_event(cls, fn: Callable[[Any], Any]):
        def wrapped(self: "EventsMixin", event: Event):
            event_type = event.type
            
            # for unhandled events, do not save last event
            if event_type not in cls.event_order:
                return True

            if event_type == EventType.BATCH_START:
                valid = cls.last_event_type != EventType.BATCH_START

            else:
                last_event_index = cls.event_order.index(self._last_event_type)
                curr_event_index = cls.event_order.index(event_type)
                valid = last_event_index <= curr_event_index

            if valid:
                cls.last_event_type = event_type

            else:
                raise ValueError(
                    f"Lifecycle events must appear following order: {cls.event_order}. "
                    f"Instead, {cls.last_event_type} was called before {event_type}"
                )
                
            return fn(self, event)
        
        return wrapped