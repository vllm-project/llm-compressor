"""
LLM Compressor is a library for compressing large language models utilizing
the latest techniques and research in the field for both training aware and
post-training techniques.

The library is designed to be flexible and easy to use on top of
PyTorch and HuggingFace Transformers, allowing for quick experimentation.
"""

# ruff: noqa

from .logger import LoggerConfig, configure_logger, logger
from .version import __version__, version

__all__ = [
    "__version__",
    "version",
    "configure_logger",
    "logger",
    "LoggerConfig",
]

from llmcompressor.core.session_functions import (
    active_session,
    callbacks,
    create_session,
    reset_session,
)
from llmcompressor.entrypoints import Oneshot, oneshot, train
