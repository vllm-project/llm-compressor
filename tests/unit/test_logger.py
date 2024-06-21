import os

import pytest
from loguru import logger

from llmcompressor import configure_logger
from llmcompressor.logger import _logger_setup

LEVELS = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


@pytest.mark.smoke
def test_default_logger_setup():
    configure_logger()
    assert logger._core.handlers, "No handlers are set for the metrics"


@pytest.mark.smoke
def test_console_logger_setup():
    configure_logger(console_log_level="DEBUG")
    handlers = [hand for hand in logger._core.handlers.values()]
    assert handlers, "No handlers are set for the metrics"

    handler = next(
        (hand for hand in handlers if f"level={LEVELS['DEBUG']}" in str(hand)), None
    )
    assert handler is not None, "DEBUG level console handler is not set"


@pytest.mark.sanity
def test_file_logger_setup(tmp_path):
    log_file = f"{tmp_path}/test.log"
    configure_logger(log_file=log_file, log_file_level="ERROR")
    handlers = [hand for hand in logger._core.handlers.values()]
    assert handlers, "No handlers are set for the metrics"

    handler = next(
        (hand for hand in handlers if f"level={LEVELS['ERROR']}" in str(hand)), None
    )
    assert handler is not None, "ERROR level file handler is not set"


@pytest.mark.sanity
def test_env_variable_setup(monkeypatch):
    monkeypatch.setenv("LLM_COMPRESSOR_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LLM_COMPRESSOR_LOG_FILE", "env_log.json")
    monkeypatch.setenv("LLM_COMPRESSOR_LOG_FILE_LEVEL", "ERROR")

    _logger_setup(
        api_request=False,
        console_log_level=os.getenv("LLM_COMPRESSOR_LOG_LEVEL"),
        log_file=os.getenv("LLM_COMPRESSOR_LOG_FILE"),
        log_file_level=os.getenv("LLM_COMPRESSOR_LOG_FILE_LEVEL"),
    )

    handlers = [hand for hand in logger._core.handlers.values()]
    assert handlers, "No handlers are set for the metrics"

    console_handler = next(
        (hand for hand in handlers if f"level={LEVELS['WARNING']}" in str(hand)), None
    )
    assert (
        console_handler is not None
    ), "WARNING level console handler is not set from env variable"

    file_handler = next(
        (hand for hand in handlers if f"level={LEVELS['ERROR']}" in str(hand)), None
    )
    assert (
        file_handler is not None
    ), "ERROR level file handler is not set from env variable"
