# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logger configuration for Compressed Tensors.
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


def configure_logger(config: Optional[LoggerConfig] = None):
    """
    Configure the logger for Compressed Tensors.
    This function sets up the console and file logging
    as per the specified or default parameters.

    Note: Environment variables take precedence over the function parameters.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """
    logger_config = config or LoggerConfig()

    # env vars get priority
    if bool(os.getenv("COMPRESSED_TENSORS_LOG_DISABLED")):
        logger_config.disabled = True
    if bool(os.getenv("COMPRESSED_TENSORS_CLEAR_LOGGERS")):
        logger_config.clear_loggers = True
    if (console_log_level := os.getenv("COMPRESSED_TENSORS_LOG_LEVEL")) is not None:
        logger_config.console_log_level = console_log_level.upper()
    if (log_file := os.getenv("COMPRESSED_TENSORS_LOG_FILE")) is not None:
        logger_config.log_file = log_file
    if (log_file_level := os.getenv("COMPRESSED_TENSORS_LOG_FILE_LEVEL")) is not None:
        logger_config.log_file_level = log_file_level.upper()

    if logger_config.disabled:
        logger.disable("compressed_tensors")
        return

    logger.enable("compressed_tensors")

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
        log_file = logger_config.log_file or "compressed_tensors.log"
        log_file_level = logger_config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(
            log_file,
            level=log_file_level.upper(),
            serialize=True,
            filter=support_log_once,
        )


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
    message = hash(str(level) + record["message"])

    if log_once and message in _logged_once:
        return False

    if log_once:
        _logged_once.add(message)

    return True


# invoke logger setup on import with default values enabling console logging with INFO
# and disabling file logging
configure_logger(config=LoggerConfig())
