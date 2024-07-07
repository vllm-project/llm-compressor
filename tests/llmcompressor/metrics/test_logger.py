import logging
import time
from abc import ABC

import pytest

from llmcompressor.metrics import (
    LambdaLogger,
    LoggerManager,
    PythonLogger,
    SparsificationGroupLogger,
    WANDBLogger,
)


@pytest.mark.parametrize(
    "logger",
    [
        PythonLogger(),
        LambdaLogger(
            lambda_func=lambda tag, value, values, step, wall_time, level: logging.info(
                f"{tag}, {value}, {values}, {step}, {wall_time}, {level}"
            )
            or True
        ),
        *([WANDBLogger()] if WANDBLogger.available() else []),
        SparsificationGroupLogger(
            lambda_func=lambda tag, value, values, step, wall_time, level: logging.info(
                f"{tag}, {value}, {values}, {step}, {wall_time}, {level}"
            )
            or True,
            python=True,
            tensorboard=True,
            wandb_=True,
        ),
        LoggerManager(),
        LoggerManager(
            [
                WANDBLogger() if WANDBLogger.available() else PythonLogger(),
            ]
        ),
    ],
)
class TestModifierLogger(ABC):
    def test_name(self, logger):
        assert logger.name is not None

    def test_log_hyperparams(self, logger):
        logger.log_hyperparams({"param1": 0.0, "param2": 1.0})
        logger.log_hyperparams({"param1": 0.0, "param2": 1.0}, level=10)

    def test_log_scalar(self, logger):
        logger.log_scalar("test-scalar-tag", 0.1)
        logger.log_scalar("test-scalar-tag", 0.1, 1)
        logger.log_scalar("test-scalar-tag", 0.1, 2, time.time() - 1)
        logger.log_scalar("test-scalar-tag", 0.1, 2, time.time() - 1, level=10)

    def test_log_scalars(self, logger):
        logger.log_scalars("test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0})
        logger.log_scalars("test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0}, 1)
        logger.log_scalars(
            "test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0}, 2, time.time() - 1
        )
        logger.log_scalars(
            "test-scalars-tag",
            {"scalar1": 0.0, "scalar2": 1.0},
            2,
            time.time() - 1,
            level=10,
        )
