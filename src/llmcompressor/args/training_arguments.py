"""
Training argument classes for LLM compression workflows.

This module defines dataclass-based argument containers for configuring
training and one-shot calibration workflows. Extends HuggingFace's
TrainingArguments with additional parameters specific to compression and
stage-based execution.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArgs

__all__ = [
    "TrainingArguments",
]


@dataclass
class TrainingArguments(HFTrainingArgs):
    """
    Training arguments specific to LLM Compressor Transformers workflow using
    HFTrainingArgs as base class

    """

    do_oneshot: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run one-shot calibration in stages"},
    )
    run_stages: Optional[bool] = field(
        default=False, metadata={"help": "Whether to trigger recipe stage by stage"}
    )
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "The output directory where the model safetensors, "
            "recipe, config, and optionally checkpoints will be written."
        },
    )

    @property
    def place_model_on_device(self):
        return False
