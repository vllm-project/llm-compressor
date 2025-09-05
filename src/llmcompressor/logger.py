"""
Provides a flexible logging configuration for LLM Compressor.

Using the loguru library, Logger supports console and file logging with options
to configure via environment variables or direct function calls.

**Environment Variables**

- `LLM_COMPRESSOR_LOG_DISABLED`: Disable logging.
    Default: `False`.
- `LLM_COMPRESSOR_CLEAR_LOGGERS`: Clear existing loggers from loguru.
    Default: `True`.
- `LLM_COMPRESSOR_LOG_LEVEL`: Log level for console logging.
    Default: `None`. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- `LLM_COMPRESSOR_LOG_FILE`: Path to the log file for file logging.
    Default: `llm-compressor.log` if log file level set, otherwise `None`.
- `LLM_COMPRESSOR_LOG_FILE_LEVEL`: Log level for file logging.
    Default: `INFO` if log file is set, otherwise `None`.

**Usage**

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
from typing import Any, Dict, Optional

from loguru import logger

__all__ = ["LoggerConfig", "configure_logger", "logger"]


# used by `support_log_once``
_logged_once = set()


@dataclass
class LoggerConfig:
    disabled: bool = False
    clear_loggers: bool = True
    console_log_level: Optional[str] = "INFO"
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None
    metrics_disabled: bool = False


def configure_logger(config: Optional[LoggerConfig] = None) -> None:
    """
    Configure the logger for LLM Compressor.

    This function sets up the console and file logging
    as per the specified or default parameters.

    **Note**: Environment variables take precedence over function parameters.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """
    logger_config = config or LoggerConfig()

    # env vars get priority
    if (disabled := os.getenv("LLM_COMPRESSOR_LOG_DISABLED")) is not None:
        logger_config.disabled = disabled.lower() == "true"
    if (clear_loggers := os.getenv("LLM_COMPRESSOR_CLEAR_LOGGERS")) is not None:
        logger_config.clear_loggers = clear_loggers.lower() == "true"
    if (console_log_level := os.getenv("LLM_COMPRESSOR_LOG_LEVEL")) is not None:
        logger_config.console_log_level = console_log_level.upper()
    if (log_file := os.getenv("LLM_COMPRESSOR_LOG_FILE")) is not None:
        logger_config.log_file = log_file
    if (log_file_level := os.getenv("LLM_COMPRESSOR_LOG_FILE_LEVEL")) is not None:
        logger_config.log_file_level = log_file_level.upper()

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
            filter=support_log_once,
        )

    if logger_config.log_file or logger_config.log_file_level:
        log_file = logger_config.log_file or "llmcompressor.log"
        log_file_level = logger_config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(
            log_file,
            level=log_file_level.upper(),
            serialize=True,
            filter=support_log_once,
        )

    if logger_config.metrics_disabled or "METRIC" in logger._core.levels.keys():
        return

    # initialize metric logger on loguru
    logger.level("METRIC", no=38, color="<yellow>", icon="📈")


def support_log_once(record: Dict[str, Any]) -> bool:
    """
    Support logging only once using `.bind(log_once=True)`

    ```
    logger.bind(log_once=False).info("This will log multiple times")
    logger.bind(log_once=False).info("This will log multiple times")
    logger.bind(log_once=True).info("This will only log once")
    logger.bind(log_once=True).info("This will only log once")  # skipped
    ```
    """
    log_once = record["extra"].get("log_once", False)
    level = getattr(record["level"], "name", "none")
    message = str(level) + record["message"]

    if log_once and message in _logged_once:
        return False

    if log_once:
        _logged_once.add(message)

    return True


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
