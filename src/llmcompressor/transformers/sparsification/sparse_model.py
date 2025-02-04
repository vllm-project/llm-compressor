import inspect
from typing import Optional

from loguru import logger
from torch.nn import Module
from transformers import AutoModelForCausalLM

__all__ = [
    "SparseAutoModelForCausalLM",
    "get_processor_name_from_model",
]


class SparseAutoModelForCausalLM:
    def from_pretrained(*args, **kwargs):
        logger.warning(
            "SparseAutoModelForCausalLM is deprecated, "
            "please use AutoModelForCausalLM"
        )
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)


def get_processor_name_from_model(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a processor/tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the processor/tokenizer shared between teacher and model
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
