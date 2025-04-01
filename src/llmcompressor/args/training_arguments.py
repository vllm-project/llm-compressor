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

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum sequence length to use during training"},
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
