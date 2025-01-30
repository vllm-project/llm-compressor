from dataclasses import fields
from typing import Any, Dict, Union

from .data_arguments import DatasetArguments
from .model_arguments import ModelArguments
from .recipe_arguments import RecipeArguments
from .training_arguments import TrainingArguments

__all__ = [
    "get_dataclass_as_dict",
]


def get_dataclass_as_dict(
    dataclass_instance: Union[
        "ModelArguments", "RecipeArguments", "DatasetArguments", "TrainingArguments"
    ],
    dataclass_class: Union[
        "ModelArguments", "RecipeArguments", "DatasetArguments", "TrainingArguments"
    ],
) -> Dict[str, Any]:
    """
    Get the dataclass instance attributes as a dict, neglicting the inherited class.
    Ex. dataclass_class=TrainingArguments will ignore HFTrainignArguments

    """
    return {
        field.name: getattr(dataclass_instance, field.name)
        for field in fields(dataclass_class)
    }
