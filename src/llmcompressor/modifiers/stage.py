from typing import List, Optional

from pydantic import BaseModel, Field

from llmcompressor.core.events import Event
from llmcompressor.core.state import State
from llmcompressor.modifiers.interface import ModifierInterface
from llmcompressor.modifiers.modifier import Modifier

__all__ = ["StageModifiers"]


class StageModifiers(ModifierInterface, BaseModel):
    """
    Represents a collection of modifiers that are applied together as a stage.

    :param modifiers: The modifiers to apply as a stage
    :param index: The index of the stage, if applicable
    :param group: The group name of the stage, if applicable
    :param applied: Flag for indicating if this stage has has already been
    applied to the model through finalization
    """

    modifiers: List["Modifier"] = Field(default_factory=list)
    index: Optional[int] = None
    group: Optional[str] = None
    applied: bool = False

    @property
    def initialized(self) -> bool:
        """
        :return: True if all of the stage modifiers have been initialized,
            False otherwise
        """
        return all(mod.initialized for mod in self.modifiers)

    @property
    def finalized(self) -> bool:
        """
        :return: True if all of the stage modifiers have been finalized,
            False otherwise
        """
        return all(mod.finalized for mod in self.modifiers)

    @property
    def unique_id(self) -> str:
        """
        :return: ID for stage containing the name and index
        """
        return self.group + "_" + str(self.index)

    def calculate_start(self) -> float:
        """
        :return: The minimum start time of all the stage modifiers
        """
        return min(
            mod.calculate_start()
            for mod in self.modifiers
            if mod.calculate_start() >= 0
        )

    def calculate_end(self) -> float:
        """
        :return: The maximum end time of all the stage modifiers, or -1 if none of the
        modifiers have set ends
        """
        return max(mod.calculate_end() for mod in self.modifiers)

    def initialize(self, state: "State", **kwargs):
        """
        Initialize all the stage modifiers

        :param state: The state of current session
        :param kwargs: Additional kwargs to pass to the modifier(s)
            initialize method
        """

        if self.applied:
            return

        accelerator = kwargs.get("accelerator", None)
        for modifier in self.modifiers:
            modifier.initialize(state, **kwargs)
            if accelerator:
                accelerator.wait_for_everyone()
        state.loggers.system.info(tag="stage", string="Modifiers initialized")

    def finalize(self, state: "State", **kwargs):
        """
        Finalize all the stage modifiers and mark the stage as applied

        :param state: The state of current session
        :param kwargs: Additional kwargs to pass to the modifier(s)
            finalize method
        """

        if self.applied:
            return

        accelerator = kwargs.get("accelerator", None)
        for modifier in self.modifiers:
            modifier.finalize(state, **kwargs)
            if accelerator:
                accelerator.wait_for_everyone()

        self.applied = True
        state.loggers.system.info(tag="stage", string="Modifiers finalized")

    def update_event(self, state: "State", event: "Event", **kwargs):
        """
        Propagate the event to all the stage modifiers

        :param state: The state of current session
        :param event: The event to propagate
        :param kwargs: Additional kwargs to pass to the modifier(s)
            update_event method
        """

        if self.applied:
            return

        for modifier in self.modifiers:
            modifier.update_event(state, event, **kwargs)
