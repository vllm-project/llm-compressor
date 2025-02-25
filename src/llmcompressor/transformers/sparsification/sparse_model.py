from loguru import logger
from transformers import AutoModelForCausalLM

__all__ = [
    "SparseAutoModelForCausalLM",
]


class SparseAutoModelForCausalLM:
    def from_pretrained(*args, **kwargs):
        logger.warning(
            "SparseAutoModelForCausalLM is deprecated, "
            "please use AutoModelForCausalLM"
        )
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)
