from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HFTrainingArgs

__all__ = ["TrainingArguments"]


@dataclass
class TrainingArguments(HFTrainingArgs):
    """
    Training arguments specific to LLM Compressor Transformers workflow

    :param best_model_after_epoch (`int`, *optional*, defaults to None):
        The epoch after which best model will be saved; used in conjunction
        with `load_best_model_at_end` and `metric_for_best_model` training
        arguments
    """

    recipe: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a LLM Compressor sparsification recipe",
        },
    )
    recipe_args: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of recipe arguments to evaluate, of the format key1=value1 "
                "key2=value2"
            )
        },
    )
    save_compressed: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to compress sparse models during save"},
    )
    do_oneshot: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run one-shot calibration"},
    )
    run_stages: Optional[bool] = field(
        default=False, metadata={"help": "Whether to trigger recipe stage by stage"}
    )
    oneshot_device: Optional[str] = field(
        default="cuda:0",
        metadata={"help": "Device to run oneshot calibration on"},
    )
    clear_sparse_session: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to clear CompressionSession data between runs."},
    )
    save_safetensors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of "
            "default torch.load and torch.save."
        },
    )
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "The output directory where the model predictions and "
            "checkpoints will be written."
        },
    )

    @property
    def place_model_on_device(self):
        return False
