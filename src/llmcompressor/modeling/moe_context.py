"""
Standardized interface for MoE model calibration.
MoE calibration context is used to apply MoE calibration modifications to the model.
There are two types of MoE calibration contexts:
1. ContextualMoECalibration: uses context managers for temporary modifications
    and restores the model to its original state after pipeline execution
2. PermanentMoECalibration: permanently modifies the model
"""

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar, Union

import tqdm
from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.utils.helpers import patch_attr

T = TypeVar("T", bound="MoECalibrationContext")


class MoECalibrationType(Enum):
    """Enumeration of supported MoE calibration types."""

    PERMANENT = "permanent"
    CONTEXTUAL = "contextual"


@dataclass
class MoEModelConfig:
    """
    Configuration for MoE model calibration.

    This dataclass defines the parameters needed to configure MoE calibration
    for a specific model architecture. It follows the same pattern used by
    other model configuration systems in the project (e.g., SmoothQuant, AWQ).

    Attributes:
        calibration_type: Type of calibration - MoECalibrationType.PERMANENT or
            MoECalibrationType.CONTEXTUAL
        target_class_name: The class name of the MoE module to replace
        replace_function: Function that creates the replacement module
            generally defined in modeling/model_name.py
        target_attribute: For contextual calibration, the attribute to replace
        description: Optional description of the model configuration
    """

    calibration_type: MoECalibrationType
    target_class_name: str
    replace_function: Callable
    target_attribute: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if (
            self.calibration_type == MoECalibrationType.CONTEXTUAL
            and self.target_attribute is None
        ):
            raise ValueError("target_attribute is required for contextual calibration")

        if (
            self.calibration_type == MoECalibrationType.PERMANENT
            and self.target_attribute is not None
        ):
            raise ValueError(
                "target_attribute should not be set for permanent calibration"
            )


# Registry of MoE model configurations
# Add new MoE models here following the same pattern as MAPPINGS_REGISTRY
MOE_MODEL_REGISTRY: Dict[str, MoEModelConfig] = {}


class MoECalibrationContext(ABC):
    """
    Abstract base class for MoE calibration.
    This provides a standardized interface for MoE model calibration.
    """

    @abstractmethod
    def apply(self, model: PreTrainedModel, calibrate_all_experts: bool = True) -> None:
        """
        Apply MoE calibration modifications to the model.
        :param model: The model to modify
        :param calibrate_all_experts: Whether to calibrate all
        experts or only routed ones
        """
        pass

    @abstractmethod
    def restore(self, model: PreTrainedModel) -> None:
        """
        Restore the model to its original state.
        :param model: The model to restore
        """
        pass


class ContextualMoECalibration(MoECalibrationContext):
    """
    MoE calibration that uses context managers for temporary modifications.
    This is suitable for models that need to be restored after calibration.
    """

    def __init__(self, model_class_name: str, update_function):
        """
        Initialize the context manager-based MoE calibration.
        :param model_class_name: The class name of the model this context applies to
        :param update_function: Function that applies the MoE modifications
        """
        self.model_class_name = model_class_name
        self.update_function = update_function
        self._stack = None

    def apply(self, model: PreTrainedModel, calibrate_all_experts: bool = True) -> None:
        """Apply MoE calibration modifications using context managers."""
        if self._stack is None:
            self._stack = contextlib.ExitStack()
            self._stack.__enter__()

        self.update_function(model, self._stack, calibrate_all_experts)

    def restore(self, model: PreTrainedModel) -> None:
        """Restore the model by exiting the context stack."""
        if self._stack is not None:
            self._stack.__exit__(None, None, None)
            self._stack = None


class PermanentMoECalibration(MoECalibrationContext):
    """
    MoE calibration context that permanently modifies the model.
    This is suitable for models that can be loaded in their modified form
    (e.g., Llama4 in vLLM).
    """

    def __init__(self, model_class_name: str, replacement_function):
        """
        Initialize the permanent MoE calibration.
        :param model_class_name: The class name of the model this context applies to
        :param replacement_function: Function that permanently replaces MoE modules
        """
        self.model_class_name = model_class_name
        self.replacement_function = replacement_function
        self._original_modules = {}

    def apply(self, model: PreTrainedModel, calibrate_all_experts: bool = True) -> None:
        """Apply permanent MoE calibration modifications."""
        # Store original modules for potential restoration
        for name, module in model.named_modules():
            if module.__class__.__name__ == self.model_class_name:
                self._original_modules[name] = module

        # Apply the replacement
        self.replacement_function(model, calibrate_all_experts)

    def restore(self, model: PreTrainedModel) -> None:
        """Restore original modules (if needed)."""
        # For permanent MoE calibrations, restoration is typically not needed
        # as the model is meant to stay in its modified form
        pass


# Registry for MoE calibrations
_MOE_CONTEXTS: Dict[str, MoECalibrationContext] = {}


def register_moe_context(model_class_name: str, context: MoECalibrationContext) -> None:
    """
    Register a MoE calibration context for a model class.
    :param model_class_name: The class name of the model
    :param context: The MoE calibration context to register
    """
    _MOE_CONTEXTS[model_class_name] = context


def get_moe_context(model_class_name: str) -> Union[MoECalibrationContext, None]:
    """
    Get the registered MoE calibration context for a model class.
    :param model_class_name: The class name of the model
    :return: The MoE calibration context or None if not found
    """
    return _MOE_CONTEXTS.get(model_class_name)


def list_supported_models() -> list:
    """
    List all model classes that have registered MoE calibration contexts.
    :return: List of supported model class names
    """
    return list(_MOE_CONTEXTS.keys())


# Generic factory functions for creating MoE updaters
def create_permanent_moe_updater(target_class_name: str, replace_function: Callable):
    """
    Create a permanent MoE updater function for the given target class.

    Args:
        target_class_name: The class name to look for in the model
        replace_function: Function that creates the replacement module

    Returns:
        A function that can be used with PermanentMoECalibration
    """

    def update_function(model: PreTrainedModel, calibrate_all_experts: bool):
        """Update MoE modules for calibration."""
        for name, module in tqdm.tqdm(list(model.named_modules())):
            if module.__class__.__name__ == target_class_name:
                new_module = replace_function(
                    config=model.config,
                    module=module,
                    calibrate_all_experts=calibrate_all_experts,
                )
                replace_module(model, name, new_module)

    return update_function


def create_contextual_moe_updater(
    target_class_name: str, target_attr: str, replace_function: Callable
):
    """
    Create a contextual MoE updater function for the given target class and attribute.

    Args:
        target_class_name: The class name to look for in the model
        target_attr: The attribute name to replace within the target class
        replace_function: Function that creates the replacement module

    Returns:
        A function that can be used with ContextualMoECalibration
    """

    def update_function(
        model: PreTrainedModel, stack: contextlib.ExitStack, calibrate_all_experts: bool
    ):
        """Update MoE modules for calibration using context managers."""
        for module in model.modules():
            if module.__class__.__name__ == target_class_name:
                stack.enter_context(
                    patch_attr(
                        module,
                        target_attr,
                        replace_function(
                            config=model.config,
                            module=getattr(module, target_attr),
                            calibrate_all_experts=calibrate_all_experts,
                        ),
                    )
                )

    return update_function


def register_moe_model(model_class_name: str, config: MoEModelConfig):
    """
    Register a MoE model with its configuration.

    Args:
        model_class_name: The model class name
        config: MoEModelConfig dataclass instance with calibration parameters
    """
    if config.calibration_type == MoECalibrationType.PERMANENT:
        updater = create_permanent_moe_updater(
            config.target_class_name, config.replace_function
        )
        context = PermanentMoECalibration(config.target_class_name, updater)
    elif config.calibration_type == MoECalibrationType.CONTEXTUAL:
        updater = create_contextual_moe_updater(
            config.target_class_name, config.target_attribute, config.replace_function
        )
        context = ContextualMoECalibration(model_class_name, updater)
    else:
        raise ValueError(f"Unknown MoE type: {config.calibration_type}")

    register_moe_context(model_class_name, context)


def register_moe_model_from_dict(model_class_name: str, config_dict: dict):
    """
    Register a MoE model from a dictionary configuration (backward compatibility).

    Args:
        model_class_name: The model class name
        config_dict: Dictionary with calibration parameters
    """
    # Convert string calibration_type to enum
    if "calibration_type" in config_dict and isinstance(
        config_dict["calibration_type"], str
    ):
        config_dict["calibration_type"] = MoECalibrationType(
            config_dict["calibration_type"]
        )

    config = MoEModelConfig(**config_dict)
    register_moe_model(model_class_name, config)
