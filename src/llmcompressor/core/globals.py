from typing import TYPE_CHECKING

from transformers import PreTrainedModel

if TYPE_CHECKING:
    from llmcompressor.core import LLMCompressor, State


def get_compressor() -> "LLMCompressor":
    from llmcompressor.core.llmcompressor import LLMCompressor

    return LLMCompressor.instance()


def get_state() -> "State":
    from llmcompressor.core.llmcompressor import LLMCompressor

    return LLMCompressor.instance().state


def get_model() -> PreTrainedModel:
    from llmcompressor.core.llmcompressor import LLMCompressor

    return LLMCompressor.instance().state.model
