"""
Logger configuration for LLM Compressor.

This module provides a flexible logging configuration using the loguru library.
It supports console and file logging with options to configure via environment
variables or direct function calls.

By default, logging is disabled for this library to ensure it does not
overwrite application logs. To enable logging, either set one of the
environment variables or call configure_logger from within the application code.

Environment Variables:
    - LLM_COMPRESSOR_LOG_LEVEL: Log level for console logging
        (default: none, options: DEBUG, INFO, WARNING, ERROR, CRITICAL).
    - LLM_COMPRESSOR_LOG_FILE: Path to the log file for file logging
        (default: llm-compressor.log if log file level set else none)
    - LLM_COMPRESSOR_LOG_FILE_LEVEL: Log level for file logging
        (default: INFO if log file set else none).

Usage:
    from llmcompressor.metrics import configure_logger

    # Configure metrics with default settings
    configure_logger()

    # Configure metrics with custom settings
    configure_logger(
        console_log_level="DEBUG",
        log_file="/path/to/logfile.log",
        log_file_level="ERROR"
    )
"""

import os
import sys
from typing import Optional

from loguru import logger

__all__ = ["configure_logger", "logger"]


def configure_logger(
    console_log_level: Optional[str] = "INFO",
    log_file: Optional[str] = None,
    log_file_level: Optional[str] = None,
):
    """
    Configure the metrics for LLM Compressor.
    This function sets up the console and file logging
    as per the specified or default parameters.

    :param console_log_level: Log level for console output, defaults to "INFO"
    :type console_log_level: Optional[str]
    :param log_file: Path to the log file, defaults to "llm-compressor.log"
        if log_file_level is set
    :type log_file: Optional[str]
    :param log_file_level: Log level for file output, defaults to "INFO"
        if log_file is set
    :type log_file_level: Optional[str]
    """
    _logger_setup(True, console_log_level, log_file, log_file_level)


def _logger_setup(
    api_request: bool,
    console_log_level: Optional[str],
    log_file: Optional[str],
    log_file_level: Optional[str],
):
    enable_logging = api_request or console_log_level or log_file or log_file_level

    if not enable_logging:
        logger.disable("llmcompressor")
        return

    logger.enable("llmcompressor")
    logger.remove()

    if console_log_level:
        # log as a human readable string with the time, function, level, and message
        logger.add(
            sys.stdout,
            level=console_log_level.upper(),
            format="{time} | {function} | {level} - {message}",
        )

    if log_file or log_file_level:
        log_file = log_file or "llm-compressor.log"
        log_file_level = log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(log_file, level=log_file_level.upper(), serialize=True)


# invoke the metrics setup on import if environment variables are set
_logger_setup(
    api_request=False,
    console_log_level=os.getenv("LLM_COMPRESSOR_LOG_LEVEL"),
    log_file=os.getenv("LLM_COMPRESSOR_LOG_FILE"),
    log_file_level=os.getenv("LLM_COMPRESSOR_LOG_FILE_LEVEL"),
)
