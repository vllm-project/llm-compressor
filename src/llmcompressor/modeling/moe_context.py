"""
Standardized interface for MoE model calibration.
MoE calibration context is used to apply MoE calibration modifications to the model.
There are two types of MoE calibration contexts:
1. ContextualMoECalibration: uses context managers for temporary modifications
2. PermanentMoECalibration: permanently modifies the model
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Union

from transformers import PreTrainedModel

T = TypeVar("T", bound="MoECalibrationContext")


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

        self.update_function(model, self._stack, calibrate_all_experts)

    def restore(self, model: PreTrainedModel) -> None:
        """Restore the model by exiting the context stack."""
        if self._stack is not None:
            self._stack.close()
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


# Convenience function for backward compatibility
def create_context_manager_context(model_class_name: str, update_function):
    """
    Create a context manager-based MoE calibration.
    :param model_class_name: The class name of the model
    :param update_function: Function that applies the MoE modifications
    :return: A ContextualMoECalibration instance
    """
    return ContextualMoECalibration(model_class_name, update_function)


def create_permanent_context(model_class_name: str, replacement_function):
    """
    Create a permanent MoE calibration.
    :param model_class_name: The class name of the model
    :param replacement_function: Function that permanently replaces MoE modules
    :return: A PermanentMoECalibration instance
    """
    return PermanentMoECalibration(model_class_name, replacement_function)
