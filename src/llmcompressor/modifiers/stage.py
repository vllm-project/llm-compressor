from typing import List, Optional

from loguru import logger
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
    applied to the model, through structure initialization or finalization
    """

    modifiers: List["Modifier"] = Field(default_factory=list)
    index: Optional[int] = None
    group: Optional[str] = None
    applied: bool = False

    @property
    def initialized_structure(self) -> bool:
        """
        :return: True if any of the stage modifiers have initialized structure,
            False otherwise
        """
        return any(mod.initialized_structure for mod in self.modifiers)

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

    def check_initialized(self):
        """
        Check if all of the stage modifiers have been initialized, and log a warning
        if not. This warning is expected when loading an input recipe during finetuning
        """

        at_least_one_initialized = False
        for modifier in self.modifiers:
            if modifier.initialized:
                at_least_one_initialized = True
        if not at_least_one_initialized:
            modifier_names = [type(mod).__name__ for mod in self.modifiers]
            logger.warning(
                f"Found no initialized modifiers in stage {self.group}. "
                "Found the following uninitialized modifiers: "
                f"{modifier_names}"
            )

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

    def pre_initialize_structure(self, state: "State", **kwargs):
        """
        Pre initialize the structure for all stage modifiers mark the stage applied

        :param state: The current state of the training
        :param kwargs: Additional kwargs to pass to the modifier(s)
            pre_initialize_structure method
        """
        for modifier in self.modifiers:
            modifier.pre_initialize_structure(state, **kwargs)

        self.applied = True
        state.loggers.system.info(tag="stage", string="Model structure initialized")

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
