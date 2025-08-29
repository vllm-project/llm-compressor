"""
Enhanced trainer class for fine-tuning with compression support.

This module provides a Trainer class that extends HuggingFace's Trainer with
LLM compression session management capabilities. Integrates compression
workflows into the standard training loop for seamless model optimization
during fine-tuning.
"""

from transformers import Trainer as HFTransformersTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["Trainer"]


class Trainer(SessionManagerMixIn, HFTransformersTrainer):
    pass
