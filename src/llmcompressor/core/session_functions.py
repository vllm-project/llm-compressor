import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from llmcompressor.core.events import EventType
from llmcompressor.core.session import CompressionSession
from llmcompressor.core.state import ModifiedState
from llmcompressor.recipe import Recipe

__all__ = [
    "create_session",
    "active_session",
    "reset_session",
    "pre_initialize_structure",
    "initialize",
    "finalize",
    "apply",
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


def pre_initialize_structure(**kwargs):
    """
    A method to pre-initialize the structure of the model for the active session

    :param kwargs: the kwargs to pass to the active session's pre-initialize-structure
        method
    """
    active_session().pre_initialize_structure(**kwargs)


def initialize(
    recipe: Union[str, List[str], "Recipe", List["Recipe"], None] = None,
    recipe_stage: Union[str, List[str], None] = None,
    recipe_args: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    attach_optim_callbacks: bool = True,
    train_data: Optional[Any] = None,
    val_data: Optional[Any] = None,
    test_data: Optional[Any] = None,
    calib_data: Optional[Any] = None,
    copy_data: bool = True,
    start: Optional[float] = None,
    steps_per_epoch: Optional[int] = None,
    batches_per_step: Optional[int] = None,
    **kwargs,
) -> ModifiedState:
    """
    A method to initialize the active session for sparsification

    :param recipe: the recipe to use for the sparsification, can be a path to a
        recipe file, a raw recipe string, a recipe object, or a list of recipe objects.
    :param recipe_stage: the stage to target for the sparsification
    :param recipe_args: the args to use for overriding the recipe defaults
    :param model: the model to sparsify
    :param teacher_model: the teacher model to use for knowledge distillation
    :param optimizer: the optimizer to use for the sparsification
    :param attach_optim_callbacks: True to attach the optimizer callbacks to the
        sparsification lifecycle, False otherwise
    :param train_data: the training data to use for the sparsification
    :param val_data: the validation data to use for the sparsification
    :param test_data: the testing data to use for the sparsification
    :param calib_data: the calibration data to use for the sparsification
    :param copy_data: True to copy the data, False otherwise
    :param start: the start epoch to use for the sparsification
    :param steps_per_epoch: the number of steps per epoch to use for the
        sparsification
    :param batches_per_step: the number of batches per step to use for
        sparsification
    :param kwargs: additional kwargs to pass to the lifecycle's initialize method
    :return: the modified state of the active session after initializing
    """
    return active_session().initialize(
        recipe=recipe,
        recipe_stage=recipe_stage,
        recipe_args=recipe_args,
        model=model,
        teacher_model=teacher_model,
        optimizer=optimizer,
        attach_optim_callbacks=attach_optim_callbacks,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        calib_data=calib_data,
        copy_data=copy_data,
        start=start,
        steps_per_epoch=steps_per_epoch,
        batches_per_step=batches_per_step,
        **kwargs,
    )


def finalize(**kwargs) -> ModifiedState:
    """
    Method to finalize the active session for sparsification

    :param kwargs: additional kwargs to pass to the lifecycle's finalize method
    :return: the modified state of the active session after finalizing
    """
    return active_session().finalize(**kwargs)


def apply(
    recipe: Union[str, List[str], "Recipe", List["Recipe"], None] = None,
    recipe_stage: Union[str, List[str], None] = None,
    recipe_args: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    train_data: Optional[Any] = None,
    val_data: Optional[Any] = None,
    test_data: Optional[Any] = None,
    calib_data: Optional[Any] = None,
    copy_data: bool = True,
    start: Optional[float] = None,
    steps_per_epoch: Optional[int] = None,
    batches_per_step: Optional[int] = None,
    **kwargs,
) -> ModifiedState:
    """
    A method to apply the recipe in one-shot manner. This will invoke the initialize
    and then finalize methods for each modifier in the active session's lifecycle.

    :param recipe: the recipe to use for the sparsification, can be a path to a
        recipe file, a raw recipe string, a recipe object, or a list of recipe objects.
    :param recipe_stage: the stage to target for the sparsification
    :param recipe_args: the args to use for overriding the recipe defaults
    :param model: the model to sparsify
    :param teacher_model: the teacher model to use for knowledge distillation
    :param train_data: the training data to use for the sparsification
    :param val_data: the validation data to use for the sparsification
    :param test_data: the testing data to use for the sparsification
    :param calib_data: the calibration data to use for the sparsification
    :param copy_data: True to copy the data, False otherwise
    :param start: the start epoch to use for the sparsification
    :param steps_per_epoch: the number of steps per epoch to use for the
        sparsification
    :param batches_per_step: the number of batches per step to use for
    :param kwargs: additional kwargs to pass to the current session's apply method
    :return: the modified state of the active session after applying the recipe
    """
    return active_session().apply(
        recipe=recipe,
        recipe_stage=recipe_stage,
        recipe_args=recipe_args,
        model=model,
        teacher_model=teacher_model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        calib_data=calib_data,
        copy_data=copy_data,
        start=start,
        steps_per_epoch=steps_per_epoch,
        batches_per_step=batches_per_step,
        **kwargs,
    )


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
        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
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
