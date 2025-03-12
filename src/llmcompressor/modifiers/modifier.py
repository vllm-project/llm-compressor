from abc import abstractmethod
from typing import Optional

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

    :param index: The index of the modifier in the list of modifiers
        for the model
    :param group: The group name for the modifier
    :param start: The start step for the modifier
    :param end: The end step for the modifier
    :param update: The update step for the modifier
    """

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

    def check_initialized(self):
        """
        :raises RuntimeError: if the modifier has not been initialized
        """
        if not self.initialized_:
            raise RuntimeError("modifier has not been initialized")

    def calculate_start(self) -> float:
        """
        Calculate and return the start epoch for the modifier.

        :return: the start epoch for the modifier if set, else -1
        """
        return self.start if self.start is not None else -1

    def calculate_end(self) -> float:
        """
        :return: the end epoch for the modifier if set, else -1
        """
        return self.end if self.end is not None else -1

    def initialize(self, state: State, **kwargs):
        """
        Initialize the modifier for the given model and state.

        :raises RuntimeError: if the modifier has already been finalized
        :param state: The current state of the model
        :param kwargs: Additional arguments for initializing the modifier
        """
        if self.initialized_:
            return

        if self.finalized_:
            raise RuntimeError("cannot initialize a finalized modifier")

        if state.start_event is None:
            return

        # ignore modifier structure initialized from one-shot
        if state.start_event.current_index >= 0 and self.calculate_start() < 0:
            return

        # if modifier should have ended by current index, don't initialize
        if (
            self.calculate_end() >= 0
            and state.start_event.current_index >= self.calculate_end()
        ):
            return

        initialized = self.on_initialize(state=state, **kwargs)

        if not isinstance(initialized, bool):
            raise ValueError(
                "on_initialize must return a boolean value; "
                "True for success, False for not initialized"
            )

        self.initialized_ = initialized

        if self.should_start(state.start_event):
            self.on_start(state, state.start_event, **kwargs)
            self.started_ = True

    def finalize(self, state: State, **kwargs):
        """
        Finalize the modifier for the given model and state.

        :raises RuntimeError: if the modifier has not been initialized
        :param state: The current state of the model
        :param kwargs: Additional arguments for finalizing the modifier
        """
        if self.finalized_ or not self.initialized_:
            return

        if not self.initialized_:
            raise RuntimeError("cannot finalize an uninitialized modifier")

        finalized = self.on_finalize(state=state, **kwargs)

        if not isinstance(finalized, bool):
            raise ValueError(
                "on_finalize must return a boolean value; "
                "True for success, False for not finalized"
            )

        self.finalized_ = finalized

    def update_event(self, state: State, event: Event, **kwargs):
        """
        Update modifier based on the given event. In turn calls
        on_start, on_event, and on_end based on the event and
        modifier settings. Returns immediately if the modifier is
        not initialized

        :raises RuntimeError: if the modifier has been finalized
        :param state: The current state of sparsification
        :param event: The event to update the modifier with
        :param kwargs: Additional arguments for updating the modifier
        """
        if not self.initialized_:
            return

        if self.finalized_:
            raise RuntimeError("cannot update a finalized modifier")

        # handle starting the modifier if needed
        if (
            event.type_ == EventType.BATCH_START
            and not self.started_
            and self.should_start(event)
        ):
            self.on_start(state, event, **kwargs)
            self.started_ = True

        if self.started_ and not self.ended_:
            self.on_event(state, event, **kwargs)

        # handle ending the modifier if needed
        if (
            event.type_ == EventType.BATCH_END
            and not self.ended_
            and self.should_end(event)
        ):
            self.on_end(state, event, **kwargs)
            self.ended_ = True

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
