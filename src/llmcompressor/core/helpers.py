"""
Helper functions for core compression operations.

Provides utility functions for logging model information and state
management during compression workflows. Includes functionality for
conditional logging and parameter tracking.
"""

from typing import Any, Generator, Optional, Tuple, Union

from llmcompressor.core.state import State
from llmcompressor.metrics import LoggerManager

__all__ = [
    "should_log_model_info",
    "log_model_info",
]


def should_log_model_info(
    model: Any,
    loggers: LoggerManager,
    current_log_step: float,
    last_log_step: Optional[float] = None,
) -> bool:
    """
    Check if we should log model level info
    Criteria:
        - model has a loggable_items method
        - state has a metrics manager
        - metrics manager is ready to log based on cadence and last log epoch

    :param model: The model whose info we want to log
    :param loggers: The metrics manager to log to
    :param current_log_step: The current epoch
    :param last_log_step: The last step we logged model info at
    :return: True if we should log model level info, False otherwise
    """
    return (
        hasattr(model, "loggable_items")
        and isinstance(loggers, LoggerManager)
        and loggers.log_ready(
            current_log_step=current_log_step, last_log_step=last_log_step
        )
    )


def log_model_info(state: State, current_log_step):
    """
    Log model level info to the metrics
    Relies on `state.model` having a `loggable_items` method
    that returns a generator of tuples of the loggable item
    name and value. Also relies on `state.loggers` being a
    `LoggerManager` instance.

    :param state: The current state of sparsification
    :param current_log_step: The current log step to log
        model info at
    """
    _log_current_step(logger_manager=state.loggers, current_log_step=current_log_step)
    _log_model_loggable_items(
        logger_manager=state.loggers,
        loggable_items=state.model.loggable_items(),
        epoch=current_log_step,
    )


def _log_current_step(
    logger_manager: LoggerManager, current_log_step: Union[float, int]
):
    """
    Log the Current Log Step to the logger_manager

    :param logger_manager: The metrics manager to log to
    :param current_log_step: The logging step
    """
    tag = logger_manager.frequency_manager.frequency_type
    logger_manager.log_scalar(tag=tag, value=current_log_step, step=current_log_step)


def _log_model_loggable_items(
    logger_manager: LoggerManager,
    loggable_items: Generator[Tuple[str, Any], None, None],
    epoch: float,
):
    """
    Log the model level loggable items to the logger_manager

    :param logger_manager: The metrics manager to log to
    :param loggable_items: The loggable items to log, must be a generator of tuples
        of the loggable item name and value
    :param epoch: The epoch to log
    """
    for loggable_item in loggable_items:
        log_tag, log_value = loggable_item
        if isinstance(log_value, dict):
            logger_manager.log_scalars(tag=log_tag, values=log_value, step=epoch)
        elif isinstance(log_value, (int, float)):
            logger_manager.log_scalar(tag=log_tag, value=log_value, step=epoch)
        else:
            logger_manager.log_string(tag=log_tag, string=log_value, step=epoch)
