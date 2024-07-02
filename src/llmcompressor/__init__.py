"""
A library for compressing large language models utilizing the latest techniques and
research in the field for both training aware and post training techniques.

The library is designed to be flexible and easy to use on top of
PyTorch and HuggingFace Transformers, allowing for quick experimentation.
"""

# flake8: noqa

from .logger import LoggerConfig, configure_logger, logger
from .version import (
    __version__,
    build_type,
    version,
    version_base,
    version_build,
    version_major,
    version_minor,
    version_patch,
)

__all__ = [
    "__version__",
    "version_base",
    "build_type",
    "version",
    "version_major",
    "version_minor",
    "version_patch",
    "version_build",
    "configure_logger",
    "logger",
    "LoggerConfig",
]

from llmcompressor.core.session_functions import (
    active_session,
    apply,
    callbacks,
    create_session,
    finalize,
    initialize,
    pre_initialize_structure,
    reset_session,
)
