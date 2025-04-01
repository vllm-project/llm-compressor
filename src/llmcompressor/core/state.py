"""
Module for managing the state of the LLM Compressor.

This module provides classes for holding and updating the state information
related to data, hardware, and model compression.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel

from llmcompressor.typing import Processor

__all__ = ["State"]


@dataclass
class State:
    """
    State class holds information about the current compression state.

    :param model: The model being used for compression
    :type model: Any
    :param teacher_model: The teacher model being used for compression
    :type teacher_model: Any
    :param optimizer: The optimizer being used for training
    :type optimizer: Any
    """

    model: PreTrainedModel
    processor: Processor
    current_index: int = -1

    teacher_model: Optional[PreTrainedModel] = None  # TODO: likely unnecessary as state
    loss: Optional[torch.Tensor] = None
    optimizer: Optional[torch.optim.Optimizer] = None
