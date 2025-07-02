"""
Module for managing the compression lifecycle in the LLM Compressor.

This module provides a class for defining and managing the lifecycle of compression
events, including initialization, finalization, and event handling.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from loguru import logger

from llmcompressor.core.events import Event, EventType
from llmcompressor.core.state import State
from llmcompressor.recipe import Recipe, RecipeArgsInput, RecipeInput, RecipeStageInput

__all__ = ["CompressionLifecycle"]


@dataclass
class CompressionLifecycle:
    """
    A class for managing the lifecycle of compression events in the LLM Compressor.

    :param state: The current state of the compression process
    :type state: Optional[State]
    :param recipe: The compression recipe
    :type recipe: Recipe
    :param modifiers: The list of stage modifiers
    :type modifiers: List[StageModifiers]
    """

    state: State = field(default_factory=State)
    recipe: Recipe = field(default_factory=Recipe)

    initialized_: bool = False
    finalized: bool = False

    # event order validation
    _last_event_type: Optional[EventType] = EventType.BATCH_END
    _event_order: List[EventType] = field(
        default_factory=lambda: [
            EventType.BATCH_START,
            EventType.LOSS_CALCULATED,
            EventType.OPTIM_PRE_STEP,
            EventType.OPTIM_POST_STEP,
            EventType.BATCH_END,
        ]
    )

    # track global step in training (could be epoch/batch)
    global_step: int = 0

    def reset(self):
        """
        Reset the compression lifecycle, finalizing any active modifiers
        and resetting all attributes.
        """
        logger.debug("Resetting compression lifecycle")

        for mod in self.recipe.modifiers:
            if not mod.initialized or mod.finalized:
                continue
            try:
                mod.finalize(self.state)
                logger.debug("Finalized modifier: {}", mod)
            except Exception as e:
                logger.warning(f"Exception during finalizing modifier: {e}")

        self.__init__()
        logger.info("Compression lifecycle reset")

    def initialize(
        self,
        recipe: Optional[RecipeInput] = None,
        recipe_stage: Optional[RecipeStageInput] = None,
        recipe_args: Optional[RecipeArgsInput] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Initialize the compression lifecycle.

        :param kwargs: Additional arguments to update the state with
        :return: List of data returned from initialization of modifiers
        :rtype: List[Any]
        """

        self.state.update(**kwargs)
        if self.initialized_:  # TODO: do not initialize twice
            return

        logger.debug("Initializing compression lifecycle")
        if not recipe:
            self.recipe = Recipe()
        else:
            self.recipe = Recipe.create_instance(
                path_or_modifiers=recipe, target_stage=recipe_stage
            )
            if recipe_args:
                self.recipe.args = {**recipe_args}

        mod_data = []
        for mod in self.recipe.modifiers:
            data = mod.initialize(state=self.state, **kwargs)
            logger.debug("Initialized modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        self.initialized_ = True
        logger.info(
            "Compression lifecycle initialized for {} modifiers",
            len(self.recipe.modifiers),
        )

        return mod_data

    def finalize(self, **kwargs) -> List[Any]:
        """
        Finalize the compression lifecycle.

        :param kwargs: Additional arguments to update the state with
        :return: List of data returned from finalizing modifiers
        :rtype: List[Any]
        :raises ValueError: If called before initialization or more than once
        """
        if not self.initialized_:
            logger.error("Cannot finalize before initializing")
            raise ValueError("Cannot finalize before initializing")

        if self.finalized:
            logger.error("Cannot finalize more than once")
            raise ValueError("Cannot finalize more than once")

        logger.debug("Finalizing compression lifecycle")
        mod_data = []
        for mod in self.recipe.modifiers:
            data = mod.finalize(state=self.state, **kwargs)
            logger.debug("Finalized modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        self.finalized = True

        logger.info(
            "Compression lifecycle finalized for {} modifiers",
            len(self.recipe.modifiers),
        )

        return mod_data

    def event(
        self, event_type: EventType, global_step: Optional[int] = 0, **kwargs
    ) -> List[Any]:
        """
        Handle a compression event.

        :param event_type: The type of event to handle
        :type event_type: EventType
        :param kwargs: Additional arguments to pass to the event handlers
        :return: List of data returned from handling the event by modifiers
        :rtype: List[Any]
        :raises ValueError: If called before initialization, after finalization,
            or for an invalid event type
        """
        if not self.initialized_:
            logger.error("Cannot invoke event before initializing")
            raise ValueError("Cannot invoke event before initializing")

        if self.finalized:
            logger.error("Cannot invoke event after finalizing")
            raise ValueError("Cannot invoke event after finalizing")

        if event_type in [EventType.INITIALIZE, EventType.FINALIZE]:
            logger.error(
                "Cannot invoke {} event. Use the corresponding method instead.",
                event_type,
            )
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        if not self._validate_event_order(event_type):
            raise ValueError(
                f"Lifecycle events must appear following order: {self._event_order}. "
                f"Instead, {self._last_event_type} was called before {event_type}"
            )

        if event_type == EventType.LOSS_CALCULATED and (
            "loss" not in kwargs or kwargs["loss"] is None
        ):
            logger.error("Loss must be provided for loss calculated event")
            raise ValueError("Loss must be provided for loss calculated event")

        logger.debug("Handling event: {}", event_type)

        # update global step
        if global_step is not None:
            self.global_step = global_step

        event = Event(type_=event_type)
        mod_data = []
        for mod in self.recipe.modifiers:
            data = mod.update_event(state=self.state, event=event, **kwargs)
            logger.debug("Updated event with modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        assert (
            event is not None
        ), f"Event lifecycle did not return an event for {event_type}"

        return mod_data

    def _validate_event_order(self, event_type: EventType) -> bool:
        if event_type not in self._event_order:
            # for unhandled events, do not save last event
            return True

        if event_type == EventType.BATCH_START:
            valid = self._last_event_type != EventType.BATCH_START

        else:
            last_event_index = self._event_order.index(self._last_event_type)
            curr_event_index = self._event_order.index(event_type)
            valid = last_event_index <= curr_event_index

        if valid:
            self._last_event_type = event_type
        return valid
