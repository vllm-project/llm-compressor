import math
from abc import abstractmethod
from typing import Optional

from pydantic import Field

from llmcompressor.core import Event, State
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["Modifier"]


class Modifier(HooksMixin):
    """
    A base class for all modifiers to inherit from.
    Modifiers are used to modify the training process for a model.
    Defines base attributes and methods available to all modifiers

    :param start: The start step for the modifier
    :param end: The end step for the modifier
    :param update: The update step for the modifier
    """

    start: Optional[float] = None
    end: Optional[float] = None

    initialized_: bool = Field(default=False, repr=False)
    finalized_: bool = Field(default=False, repr=False)
    started_: bool = Field(default=False, repr=False)
    ended_: bool = Field(default=False, repr=False)

    def should_start(self, state: State) -> bool:
        """
        :param event: The event to check if the modifier should start
        :return: True if the modifier should start based on the given event
        """
        start = self.start if self.start is not None else -1
        end = self.end if self.end is not None else math.inf
        current = state.current_index

        return start <= current < end  # always true for oneshot

    def should_end(self, state: State) -> bool:
        """
        :param event: The event to check if the modifier should end
        :return: True if the modifier should end based on the given event
        """
        end = self.end if self.end is not None else math.inf
        current = state.current_index

        return current >= end  # always true for oneshot

    @abstractmethod
    def on_initialize(self, state: State) -> bool:
        """
        on_initialize is called on modifier initialization and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param kwargs: Additional arguments for initializing the modifier
        :return: True if the modifier was initialized successfully,
            False otherwise
        """
        raise NotImplementedError()

    def on_start(self, state: State):
        """
        on_start is called when the modifier starts and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param event: The event that triggered the start
        :param kwargs: Additional arguments for starting the modifier
        """
        self.started_ = True

    def on_event(self, state: State, event: Event):
        """
        on_event is called whenever an event is triggered

        :param state: The current state of the model
        :param event: The event that triggered the update
        :param kwargs: Additional arguments for updating the model
        """
        pass

    def on_end(self, state: State):
        """
        on_end is called when the modifier ends and must be implemented
        by the inheriting modifier.

        :param state: The current state of the model
        :param event: The event that triggered the end
        :param kwargs: Additional arguments for ending the modifier
        """
        self.ended_ = True

    def on_finalize(self, state: State):
        """
        on_finalize is called on modifier finalization and
        must be implemented by the inheriting modifier.

        :param state: The current state of the model
        :param kwargs: Additional arguments for finalizing the modifier
        :return: True if the modifier was finalized successfully,
            False otherwise
        """
        self.finalized_ = True
