"""
Logger configuration for LLM Compressor.

This module provides a flexible logging configuration using the loguru library.
It supports console and file logging with options to configure via environment
variables or direct function calls.

Environment Variables:
    - LLM_COMPRESSOR_LOG_DISABLED: Disable logging (default: false).
    - LLM_COMPRESSOR_CLEAR_LOGGERS: Clear existing loggers from loguru (default: true).
    - LLM_COMPRESSOR_LOG_LEVEL: Log level for console logging
        (default: none, options: DEBUG, INFO, WARNING, ERROR, CRITICAL).
    - LLM_COMPRESSOR_LOG_FILE: Path to the log file for file logging
        (default: llm-compressor.log if log file level set else none)
    - LLM_COMPRESSOR_LOG_FILE_LEVEL: Log level for file logging
        (default: INFO if log file set else none).

Usage:
    from llmcompressor import logger, configure_logger, LoggerConfig

    # Configure metrics with default settings
    configure_logger(
        config=LoggerConfig(
            disabled=False,
            clear_loggers=True,
            console_log_level="DEBUG",
            log_file=None,
            log_file_level=None,
        )
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from loguru import logger

__all__ = ["LoggerConfig", "configure_logger", "logger"]


@dataclass
class LoggerConfig:
    disabled: bool = False
    clear_loggers: bool = True
    console_log_level: Optional[str] = "INFO"
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None
    metrics_disabled: bool = False


def configure_logger(config: Optional[LoggerConfig] = None):
    """
    Configure the metrics for LLM Compressor.
    This function sets up the console and file logging
    as per the specified or default parameters.

    Note: Environment variables take precedence over the function parameters.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """
    logger_config = config or LoggerConfig()

    # env vars get priority
    if (disabled := os.getenv("LLM_COMPRESSOR_LOG_DISABLED")) is not None:
        logger_config.disabled = disabled.lower()
    if (clear_loggers := os.getenv("LLM_COMPRESSOR_CLEAR_LOGGERS")) is not None:
        logger_config.clear_loggers = clear_loggers.lower()
    if (console_log_level := os.getenv("LLM_COMPRESSOR_LOG_LEVEL")) is not None:
        logger_config.console_log_level = console_log_level.upper()
    if (log_file := os.getenv("LLM_COMPRESSOR_LOG_FILE")) is not None:
        logger_config.log_file = log_file
    if (log_file_level := os.getenv("LLM_COMPRESSOR_LOG_FILE_LEVEL")) is not None:
        logger_config.log_file_level = log_file_level

    if logger_config.disabled:
        logger.disable("llmcompressor")
        return

    logger.enable("llmcompressor")

    if logger_config.clear_loggers:
        logger.remove()

    if logger_config.console_log_level:
        # log as a human readable string with the time, function, level, and message
        logger.add(
            sys.stdout,
            level=logger_config.console_log_level.upper(),
            format="{time} | {function} | {level} - {message}",
        )

    if logger_config.log_file or logger_config.log_file_level:
        log_file = logger_config.log_file or "llmcompressor.log"
        log_file_level = logger_config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(log_file, level=log_file_level.upper(), serialize=True)

    if logger_config.metrics_disabled or "METRIC" in logger._core.levels.keys():
        return

    # initialize metric logger on loguru
    _initialize_metric_logging()


def _initialize_metric_logging() -> None:
    """
    Initalize metric logging for loguru and metric-related libraries
    Defaults to stdout
    usage:
        `logger.log("METRIC", "foo description, bar result")`

    """
    logger.level("METRIC", no=38, color="<yellow>", icon="📈")


# invoke logger setup on import with default values enabling console logging with INFO
# and disabling file logging
configure_logger(
    config=LoggerConfig(
        disabled=False,
        clear_loggers=True,
        console_log_level="INFO",
        log_file=None,
        log_file_level=None,
    )
)
