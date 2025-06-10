from abc import ABC, abstractmethod

from llmcompressor.core.events import Event
from llmcompressor.core.state import State

__all__ = ["ModifierInterface"]


class ModifierInterface(ABC):
    """
    Defines the contract that all modifiers must implement
    """

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        :return: True if the modifier has been initialized
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def finalized(self) -> bool:
        """
        :return: True if the modifier has been finalized
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, state: State, **kwargs):
        """
        Initialize the modifier

        :param state: The current state of the model
        :param kwargs: Additional keyword arguments
            for modifier initialization
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, state: State, **kwargs):
        """
        Finalize the modifier

        :param state: The current state of the model
        :param kwargs: Additional keyword arguments for
            modifier finalization
        """
        raise NotImplementedError()

    @abstractmethod
    def update_event(self, state: State, event: Event, **kwargs):
        """
        Update the modifier based on the event

        :param state: The current state of the model
        :param event: The event to update the modifier with
        :param kwargs: Additional keyword arguments for
            modifier update
        """
        raise NotImplementedError()
