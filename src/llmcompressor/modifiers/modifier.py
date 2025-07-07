from abc import abstractmethod
from typing import Optional

from pydantic import ConfigDict

from llmcompressor.core.events import Event, EventType
from llmcompressor.core.state import State
from llmcompressor.modifiers.interface import ModifierInterface
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["Modifier"]


class Modifier(ModifierInterface, HooksMixin):
    """
    A base class for all modifiers to inherit from.
    Modifiers are used to modify the training process for a model.
    Defines base attributes and methods available to all modifiers

    Lifecycle:
    1. initialize
    2. on_event ->
        * on_start if self.start <= event.current_index
        * on_end if self.end >= event.current_index
    5. finalize

    :param index: The index of the modifier in the list of modifiers
        for the model
    :param group: The group name for the modifier
    :param start: The start step for the modifier
    :param end: The end step for the modifier
    :param update: The update step for the modifier
    """

    model_config = ConfigDict(extra="forbid")

    index: Optional[int] = None
    group: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    update: Optional[float] = None

    initialized_: bool = False
    finalized_: bool = False
    started_: bool = False
    ended_: bool = False

    @property
    def initialized(self) -> bool:
        """
        :return: True if the modifier has been initialized
        """
        return self.initialized_

    @property
    def finalized(self) -> bool:
        """
        :return: True if the modifier has been finalized
        """
        return self.finalized_

    def initialize(self, state: State, **kwargs):
        """
        Initialize the modifier for the given model and state.

        :raises RuntimeError: if the modifier has already been finalized
        :param state: The current state of the model
        :param kwargs: Additional arguments for initializing the modifier
        """
        if self.initialized_:
            raise RuntimeError(
                "Cannot initialize a modifier that has already been initialized"
            )

        if self.finalized_:
            raise RuntimeError(
                "Cannot initialize a modifier that has already been finalized"
            )

        self.initialized_ = self.on_initialize(state=state, **kwargs)

        # trigger starts
        fake_start_event = Event(type_=EventType.BATCH_START, global_step=0)
        if self.should_start(fake_start_event):
            self.on_start(state, fake_start_event, **kwargs)
            self.started_ = True

    def finalize(self, state: State, **kwargs):
        """
        Finalize the modifier for the given model and state.

        :raises RuntimeError: if the modifier has not been initialized
        :param state: The current state of the model
        :param kwargs: Additional arguments for finalizing the modifier
        """
        if self.finalized_:
            raise RuntimeError("cannot finalize a modifier twice")

        if not self.initialized_:
            raise RuntimeError("cannot finalize an uninitialized modifier")

        # TODO: all finalization should succeed
        self.finalized_ = self.on_finalize(state=state, **kwargs)

    def update_event(self, state: State, event: Event, **kwargs):
        """
        Update modifier based on the given event. In turn calls
        on_start, on_update, and on_end based on the event and
        modifier settings. Returns immediately if the modifier is
        not initialized

        :raises RuntimeError: if the modifier has been finalized
        :param state: The current state of sparsification
        :param event: The event to update the modifier with
        :param kwargs: Additional arguments for updating the modifier
        """
        if not self.initialized_:
            raise RuntimeError("Cannot update an uninitialized modifier")

        if self.finalized_:
            raise RuntimeError("Cannot update a finalized modifier")

        self.on_event(state, event, **kwargs)

        # handle starting the modifier if needed
        if (
            event.type_ == EventType.BATCH_START
            and not self.started_
            and self.should_start(event)
        ):
            self.on_start(state, event, **kwargs)
            self.started_ = True
            self.on_update(state, event, **kwargs)

            return

        # handle ending the modifier if needed
        if (
            event.type_ == EventType.BATCH_END
            and not self.ended_
            and self.should_end(event)
        ):
            self.on_end(state, event, **kwargs)
            self.ended_ = True
            self.on_update(state, event, **kwargs)

            return

        if self.started_ and not self.ended_:
            self.on_update(state, event, **kwargs)

    def should_start(self, event: Event) -> bool:
        """
        :param event: The event to check if the modifier should start
        :return: True if the modifier should start based on the given event
        """
        if self.start is None:
            return False

        current = event.current_index

        return self.start <= current and (self.end is None or current < self.end)

    def should_end(self, event: Event):
        """
        :param event: The event to check if the modifier should end
        :return: True if the modifier should end based on the given event
        """
        current = event.current_index

        return self.end is not None and current >= self.end

    @abstractmethod
    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        on_initialize is called on modifier initialization and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param kwargs: Additional arguments for initializing the modifier
        :return: True if the modifier was initialized successfully,
            False otherwise
        """
        raise NotImplementedError()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        on_finalize is called on modifier finalization and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param kwargs: Additional arguments for finalizing the modifier
        :return: True if the modifier was finalized successfully,
            False otherwise
        """
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        """
        on_start is called when the modifier starts and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param event: The event that triggered the start
        :param kwargs: Additional arguments for starting the modifier
        """
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        """
        on_update is called when the model in question must be
        updated based on passed in event. Must be implemented by the
        inheriting modifier.

        :param state: The current state of the model
        :param event: The event that triggered the update
        :param kwargs: Additional arguments for updating the model
        """
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        """
        on_end is called when the modifier ends and must be implemented
        by the inheriting modifier.

        :param state: The current state of the model
        :param event: The event that triggered the end
        :param kwargs: Additional arguments for ending the modifier
        """
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        """
        on_event is called whenever an event is triggered

        :param state: The current state of the model
        :param event: The event that triggered the update
        :param kwargs: Additional arguments for updating the model
        """
        pass
