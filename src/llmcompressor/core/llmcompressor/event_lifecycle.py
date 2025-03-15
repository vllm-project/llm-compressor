from typing import Callable, Any

class EventsLifecycle:
    @classmethod
    def validate_initialize(cls, fn: Callable[[Any], Any]):
        return fn
        
    @classmethod
    def validate_finalize(cls, fn: Callable[[Any], Any]):
        return fn

    @classmethod
    def validate_event(cls, fn: Callable[[Any], Any]):
        return fn