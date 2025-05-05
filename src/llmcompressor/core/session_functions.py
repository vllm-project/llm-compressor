import threading
from contextlib import contextmanager
from typing import Any, Optional

from llmcompressor.core.events import EventType
from llmcompressor.core.session import CompressionSession
from llmcompressor.core.state import ModifiedState

__all__ = [
    "create_session",
    "active_session",
    "reset_session",
    "callbacks",
    "LifecycleCallbacks",
]


_global_session = CompressionSession()
_local_storage = threading.local()
_local_storage.session = _global_session


@contextmanager
def create_session() -> CompressionSession:
    """
    Context manager to create and yield a new session for sparsification.
    This will set the active session to the new session for the duration
    of the context.

    :return: the new session
    """
    global _local_storage
    orig_session = getattr(_local_storage, "session", None)
    new_session = CompressionSession()
    _local_storage.session = new_session
    try:
        yield new_session
    finally:
        _local_storage.session = orig_session


def active_session() -> CompressionSession:
    """
    :return: the active session for sparsification
    """
    global _local_storage
    return getattr(_local_storage, "session", _global_session)


def reset_session():
    """
    Reset the currently active session to its initial state
    """
    session = active_session()
    session._lifecycle.reset()


class LifecycleCallbacks:
    """
    A class for invoking lifecycle events for the active session
    """

    @classmethod
    def event(cls, event_type: EventType, **kwargs) -> ModifiedState:
        """
        Invoke an event for the active session

        :param event_type: the event type to invoke
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        if event_type in [EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        # skip event callbacks if no recipe was provided
        if not active_session().lifecycle.recipe_container.check_any_recipe_exists():
            return

        return active_session().event(event_type, **kwargs)

    @classmethod
    def batch_start(cls, batch_data: Optional[Any] = None, **kwargs) -> ModifiedState:
        """
        Invoke a batch start event for the active session

        :param batch_data: the batch data to use for the event
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.BATCH_START, batch_data=batch_data, **kwargs)

    @classmethod
    def loss_calculated(cls, loss: Optional[Any] = None, **kwargs) -> ModifiedState:
        """
        Invoke a loss calculated event for the active session

        :param loss: the loss to use for the event
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        # log loss if loss calculated
        active_session()._log_loss(event_type=EventType.LOSS_CALCULATED, loss=loss)
        return cls.event(EventType.LOSS_CALCULATED, loss=loss, **kwargs)

    @classmethod
    def optim_pre_step(cls, **kwargs) -> ModifiedState:
        """
        Invoke an optimizer pre-step event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.OPTIM_PRE_STEP, **kwargs)

    @classmethod
    def optim_post_step(cls, **kwargs) -> ModifiedState:
        """
        Invoke an optimizer post-step event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.OPTIM_POST_STEP, **kwargs)

    @classmethod
    def batch_end(cls, **kwargs) -> ModifiedState:
        """
        Invoke a batch end event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        active_session()._log_model_info()
        return cls.event(EventType.BATCH_END, **kwargs)


callbacks = LifecycleCallbacks
