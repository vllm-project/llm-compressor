"""
Module for managing the compression lifecycle in the LLM Compressor.

This module provides a class for defining and managing the lifecycle of compression
events, including initialization, finalization, and event handling.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from loguru import logger

from llmcompressor.core.events import (
    CallbacksEventLifecycle,
    EventLifecycle,
    EventType,
    OptimizerEventLifecycle,
)
from llmcompressor.core.state import State
from llmcompressor.modifiers import StageModifiers
from llmcompressor.recipe import RecipeContainer

__all__ = ["CompressionLifecycle"]


@dataclass
class CompressionLifecycle:
    """
    A class for managing the lifecycle of compression events in the LLM Compressor.

    :param state: The current state of the compression process
    :type state: Optional[State]
    :param recipe_container: The container for the compression recipe
    :type recipe_container: RecipeContainer
    :param modifiers: The list of stage modifiers
    :type modifiers: List[StageModifiers]
    :param event_lifecycle: The event lifecycle manager
    :type event_lifecycle: Optional[EventLifecycle]
    """

    state: Optional[State] = None
    recipe_container: RecipeContainer = field(default_factory=RecipeContainer)
    modifiers: List[StageModifiers] = field(default_factory=list)
    event_lifecycle: Optional[EventLifecycle] = None

    initialized_structure: bool = False
    initialized_: bool = False
    finalized: bool = False
    event_called: bool = False

    def reset(self):
        """
        Reset the compression lifecycle, finalizing any active modifiers
        and resetting all attributes.
        """
        logger.debug("Resetting compression lifecycle")

        for mod in self.modifiers:
            if not mod.initialized or mod.finalized:
                continue
            try:
                mod.finalize(self.state)
                logger.debug("Finalized modifier: {}", mod)
            except Exception as e:
                logger.warning(f"Exception during finalizing modifier: {e}")

        self.state = None
        self.recipe_container = RecipeContainer()
        self.modifiers = []
        self.event_lifecycle = None

        self.initialized_structure = False
        self.initialized_ = False
        self.finalized = False
        self.event_called = False
        logger.info("Compression lifecycle reset")

    def pre_initialize_structure(self, **kwargs) -> List[Any]:
        """
        Pre-initialize the structure of the compression lifecycle.

        :param kwargs: Additional arguments to update the state with
        :return: List of data returned from pre-initialization of modifiers
        :rtype: List[Any]
        """
        logger.debug("Pre-initializing structure")
        self._check_create_state()
        extras = self.state.update(**kwargs)
        extras = self.recipe_container.update(**extras)

        self._check_compile_recipe()
        mod_data = []
        for mod in self.modifiers:
            data = mod.pre_initialize_structure(state=self.state, **extras)
            logger.debug("Pre-initialized modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        self.initialized_structure = True
        applied_stage_names = [mod.unique_id for mod in self.modifiers if mod.applied]
        self.recipe_container.update_applied_stages(applied_stage_names)
        logger.info(
            "Compression lifecycle structure pre-initialized for {} modifiers",
            len(self.modifiers),
        )

        return mod_data

    def initialize(self, **kwargs) -> List[Any]:
        """
        Initialize the compression lifecycle.

        :param kwargs: Additional arguments to update the state with
        :return: List of data returned from initialization of modifiers
        :rtype: List[Any]
        """
        logger.debug("Initializing compression lifecycle")
        self._check_create_state()
        extras = self.state.update(**kwargs)
        extras = self.recipe_container.update(**extras)

        self._check_compile_recipe()
        self._set_model_layer_prefix()
        mod_data = []
        for mod in self.modifiers:
            data = mod.initialize(state=self.state, **extras)
            logger.debug("Initialized modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        self.initialized_ = True
        logger.info(
            "Compression lifecycle initialized for {} modifiers", len(self.modifiers)
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
        for mod in self.modifiers:
            data = mod.finalize(state=self.state, **kwargs)
            logger.debug("Finalized modifier: {}", mod)
            if data is not None:
                mod_data.append(data)

        self.finalized = True
        applied_stage_names = [mod.unique_id for mod in self.modifiers if mod.applied]
        self.recipe_container.update_applied_stages(applied_stage_names)

        logger.info(
            "Compression lifecycle finalized for {} modifiers", len(self.modifiers)
        )

        return mod_data

    def event(self, event_type: EventType, **kwargs) -> List[Any]:
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

        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            logger.error(
                "Cannot invoke {} event. Use the corresponding method instead.",
                event_type,
            )
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        if event_type == EventType.LOSS_CALCULATED and (
            "loss" not in kwargs or kwargs["loss"] is None
        ):
            logger.error("Loss must be provided for loss calculated event")
            raise ValueError("Loss must be provided for loss calculated event")

        logger.debug("Handling event: {}", event_type)
        self._check_setup_event_lifecycle(event_type)

        event = None
        mod_data = []
        for event in self.event_lifecycle.events_from_type(event_type):
            if self.state.start_event is None:
                self.state.start_event = event

            for mod in self.modifiers:
                data = mod.update_event(state=self.state, event=event, **kwargs)
                logger.debug("Updated event with modifier: {}", mod)
                if data is not None:
                    mod_data.append(data)

        assert (
            event is not None
        ), f"Event lifecycle did not return an event for {event_type}"
        self.state.last_event = event
        self.event_called = True

        return mod_data

    def _check_create_state(self):
        if self.state is not None:
            return

        logger.debug("Creating new State instance for compression lifecycle")
        self.state = State()
        logger.info("State created for compression lifecycle")

    def _check_compile_recipe(self):
        if not self.recipe_container.check_compile_recipe():
            return

        logger.debug(
            "Compiling recipe and creating modifiers for compression lifecycle"
        )
        self.modifiers = self.recipe_container.compiled_recipe.create_modifier()
        for mod in self.modifiers:
            if mod.unique_id in self.recipe_container.applied_stages:
                mod.applied = True
        logger.info(
            "Recipe compiled and {} modifiers created",
            len(self.modifiers),
        )

    def _check_setup_event_lifecycle(self, event_type: EventType):
        if self.event_lifecycle is not None:
            return

        if (
            self.state is None
            or self.state.model is None
            or self.state.start_event is None
            or self.recipe_container.compiled_recipe is None
        ):
            logger.error("Cannot invoke event before recipe, model, and start are set")
            raise ValueError(
                "Cannot invoke event before recipe, model, and start are set"
            )

        if not self.state.compression_ready:
            logger.error("Cannot invoke event before recipe, model, and start are set")
            raise ValueError(
                "Cannot invoke event before recipe, model, and start are set"
            )

        logger.debug("Setting up event lifecycle for event type: {}", event_type)

        for mod in self.modifiers:
            logger.debug("Checking if modifier is initialized: {}", mod)
            mod.check_initialized()

        # first check for creation of a callbacks event lifecycle
        # must start with BATCH_START event
        if event_type == EventType.BATCH_START:
            self.event_lifecycle = CallbacksEventLifecycle(
                type_first=EventType.BATCH_START, start=self.state.start_event
            )
        elif (
            event_type == EventType.LOSS_CALCULATED
            or event_type == EventType.OPTIM_PRE_STEP
        ):
            self.event_lifecycle = OptimizerEventLifecycle(
                type_first=event_type, start=self.state.start_event
            )
        else:
            logger.error(
                "Invalid event type for initializing event lifecycle: "
                "{}. Must be BATCH_START, LOSS_CALCULATED, or OPTIM_PRE_STEP",
                event_type,
            )
            raise ValueError(
                f"Invalid event type for initializing event lifecycle: "
                f"{event_type}. Must be BATCH_START, LOSS_CALCULATED, or OPTIM_PRE_STEP"
            )

        logger.info(
            "Event lifecycle for compression lifecycle created: "
            "{} with start event type: {}",
            self.event_lifecycle,
            event_type,
        )

    def _set_model_layer_prefix(self):
        compiled_recipe = self.recipe_container.compiled_recipe
        if (
            compiled_recipe is None
            or (metadata := compiled_recipe.metadata) is None
            or (model_metadata := metadata.target_model) is None
        ):
            return False

        self.state.model.layer_prefix = model_metadata.layer_prefix
        logger.debug("Model layer prefix set to {}", self.state.model.layer_prefix)
        return True
