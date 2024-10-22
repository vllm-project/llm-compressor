import inspect
from pathlib import Path
from typing import Optional, Union

import torch
from torch.nn import Module
from transformers import AutoModelForCausalLM

from llmcompressor.pytorch.model_load.helpers import initialize_recipe
from llmcompressor.transformers.utils.helpers import resolve_recipe

__all__ = [
    "SparseAutoModel",
    "get_shared_tokenizer_src",
]


class SparseAutoModel:
    """
    Factory class for creating sparse models using transformers AutoModel classes
    """

    @staticmethod
    def text_generation_from_pretrained(
        model_name_or_path: str,
        sequence_length: Optional[int] = None,
        recipe: Optional[Union[str, Path]] = None,
        trust_remote_code: bool = False,
        torch_dtype: Union[str, torch.dtype] = "auto",
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param sequence_length: the maximum length of the sequence to generate.
            If None, will use the default sequence length for the model.
            Defaults to None.
        :param recipe: the recipe to apply to the model. If None, no recipe is applied
        :param trust_remote_code: related to trust_remote_code in HF transformers.
            If True, will execute the modelling code from the model directory
            (if present). Defaults to False.
        :param torch_dtype: the torch dtype to use for the model. If "auto", will
            use the default dtype for the model. Defaults to "auto".
        :return: the created model for text generation
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        recipe = resolve_recipe(recipe=recipe, model_path=model_name_or_path)

        if recipe:
            initialize_recipe(model=model, recipe_path=recipe)
        if sequence_length is not None:
            model.seqlen = sequence_length

        return model


def get_shared_tokenizer_src(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the tokenizer shared between teacher and model
    """

    if teacher is not None and teacher not in ("disable", "self"):
        student_forward_params = list(
            inspect.signature(student.forward).parameters.keys()
        )
        teacher_forward_params = list(
            inspect.signature(teacher.forward).parameters.keys()
        )
        diff = [p for p in student_forward_params if p not in teacher_forward_params]
        if diff:
            raise RuntimeError(
                "Teacher tokenizer cannot be used for student "
                f"due to missing args: {diff}"
            )
        src_model = teacher
    else:
        src_model = student
    return src_model.config._name_or_path
