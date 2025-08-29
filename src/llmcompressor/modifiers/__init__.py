"""
Compression modifiers for applying various optimization techniques.

Provides the core modifier system for applying compression techniques like
quantization, pruning, distillation, and other optimization methods to neural
networks. Includes base classes, factory patterns, and interfaces for
extensible compression workflows.
"""

from .factory import ModifierFactory
from .interface import ModifierInterface
from .modifier import Modifier

__all__ = [
    "ModifierFactory",
    "ModifierInterface",
    "Modifier",
]
