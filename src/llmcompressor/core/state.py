"""
Module for managing the state of the LLM Compressor.

This module provides classes for holding and updating the state information
related to data, hardware, and model compression.
"""

from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedModel
from llmcompressor.typing import Processor

__all__ = ["State"]


@dataclass
class State:
    """
    State class holds information about the current compression state.

    :param model: The model being used for compression
    :param processor: Processor used for data processing
    :param teacher: The teacher model being used for compression
    :param current_index: Current global step, as set by batch start events

    :type model: PreTrainedModel
    :type processor: Processor
    :type teacher_model: Optional[PreTrainedModel]
    :type current_index: int
    """

    model: PreTrainedModel
    processor: Processor
    teacher: Optional[PreTrainedModel] = None
    current_index: int = -1  # begin at -1 before 0th batch