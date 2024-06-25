"""
Contains code for loggers that help visualize the information from each modifier
"""

import logging
import os
import time
import warnings
from abc import ABC
from contextlib import contextmanager
from datetime import datetime
from logging import CRITICAL, DEBUG, ERROR, INFO, WARN, Logger
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

from llmcompressor.metrics.utils import (
    FrequencyManager,
    FrequencyType,
    LoggingModeType,
    LogStepType,
)
from llmcompressor.utils import is_package_available

try:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ModuleNotFoundError, ImportError):
        from tensorboardX import SummaryWriter
    tensorboard_import_error = None
except Exception as tensorboard_err:
    SummaryWriter = object
    tensorboard_import_error = tensorboard_err


wandb_available = is_package_available("wandb")
if wandb_available:
    import wandb

    wandb_err = None
else:
    wandb = object
    wandb_err = ModuleNotFoundError(
        "`wandb` is not installed, use `pip install wandb` to log to Weights and Biases"
    )

__all__ = [
    "BaseLogger",
    "LambdaLogger",
    "PythonLogger",
    "TensorBoardLogger",
    "WANDBLogger",
    "SparsificationGroupLogger",
    "LoggerManager",
    "LOGGING_LEVELS",
]
ALL_TOKEN = "__ALL__"
LOGGING_LEVELS = {
    "debug": DEBUG,
    "info": INFO,
    "warn": WARN,
    "error": ERROR,
    "critical": CRITICAL,
}
DEFAULT_TAG = "defaul_tag"


class BaseLogger(ABC):
    """
    Base class that all modifier loggers must implement.

    :param name: name given to the metrics, used for identification
    :param enabled: True to log, False otherwise
    """

    def __init__(self, name: str, enabled: bool = True):
        self._name = name
        self._enabled = enabled

    @property
    def name(self) -> str:
        """
        :return: name given to the metrics, used for identification
        """
        return self._name

    @property
    def enabled(self) -> bool:
        """
        :return: True to log, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to log, False otherwise
        """
        self._enabled = value

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, enabled={self._enabled})"

    def log_hyperparams(self, params: Dict[str, float]) -> bool:
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        :return: True if logged, False otherwise.
        """
        return False

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        return False

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        return False

    def log_string(
        self,
        tag: str,
        string: str,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        return False

    def save(
        self,
        file_path: str,
        **kwargs,
    ) -> bool:
        """
        :param file_path: path to a file to be saved
        :param kwargs: additional arguments that a specific metrics might use
        :return: True if saved, False otherwise
        """
        return False


class LambdaLogger(BaseLogger):
    """
    Logger that handles calling back to a lambda function with any logs.

    :param lambda_func: the lambda function to call back into with any logs.
        The expected call sequence is (tag, value, values, step, wall_time) -> bool
        The return type is True if logged and False otherwise.
    :param name: name given to the metrics, used for identification;
        defaults to lambda
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        lambda_func: Callable[
            [
                Optional[str],
                Optional[Union[float, str]],
                Optional[Dict[str, float]],
                Optional[int],
                Optional[float],
                Optional[int],
            ],
            bool,
        ],
        name: str = "lambda",
        enabled: bool = True,
    ):
        super().__init__(name, enabled)
        self._lambda_func = lambda_func
        assert lambda_func, "lambda_func must be set to a callable function"

    @property
    def lambda_func(
        self,
    ) -> Callable[
        [
            Optional[str],
            Optional[Union[float, str]],
            Optional[Dict[str, float]],
            Optional[int],
            Optional[float],
            Optional[int],
        ],
        bool,
    ]:
        """
        :return: the lambda function to call back into with any logs.
            The expected call sequence is (tag, value, values, step, wall_time)
        """
        return self._lambda_func

    def log_hyperparams(
        self,
        params: Dict,
        level: Optional[int] = None,
    ) -> bool:
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        return self._lambda_func(
            tag=None,
            value=None,
            values=params,
            step=None,
            wall_time=None,
            level=level,
        )

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        level: Optional[int] = None,
    ) -> bool:
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(
            tag=tag,
            value=value,
            values=None,
            step=step,
            wall_time=wall_time,
            level=level,
        )

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        level: Optional[int] = None,
    ) -> bool:
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(
            tag=tag,
            value=None,
            values=values,
            step=step,
            wall_time=wall_time,
            level=level,
        )


class PythonLogger(LambdaLogger):
    """
    Modifier metrics that handles printing values into a python metrics instance.

    :param logger: a metrics instance to log to, if None then will create it's own
    :param log_level: default level to log any incoming data at on the logging.Logger
        instance when an explicit log level isn't provided
    :param name: name given to the metrics, used for identification;
        defaults to python
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        logger: Logger = None,
        log_level: int = None,
        name: str = "python",
        enabled: bool = True,
    ):
        self._logger = logger or self._create_default_logger(log_level=log_level)

        super().__init__(
            lambda_func=self._log_lambda,
            name=name,
            enabled=enabled,
        )

    def __getattr__(self, item):
        return getattr(self._logger, item)

    @property
    def logger(self) -> Logger:
        """
        :return: a metrics instance to log to, if None then will create it's own
        """
        return self._logger

    def _create_default_logger(self, log_level: Optional[int] = None) -> logging.Logger:
        """
        Create a default modifier metrics,
        with a file handler logging at the debug level
        and a stream handler logging to console at the specified level.

        :param log_level: logging level for the console metrics
        :return: metrics
        """
        logger = logging.getLogger(__name__)

        # Console handler, for logging high level modifier logs
        # must be created before the file handler
        # as file handler is also a stream handler
        if not any(
            isinstance(handler, logging.StreamHandler) for handler in logger.handlers
        ):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(
                log_level or logging.getLogger("llmcompressor").level
            )
            logger.addHandler(stream_handler)

        # File handler setup, for logging modifier debug statements
        if not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        ):
            base_log_path = (
                os.environ.get("NM_TEST_LOG_DIR")
                if os.environ.get("NM_TEST_MODE")
                else "sparse_logs"
            )
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")
            log_path = os.path.join(base_log_path, f"{dt_string}.log")
            os.makedirs(base_log_path, exist_ok=True)
            file_handler = logging.FileHandler(
                log_path,
                delay=True,
            )
            file_handler.setLevel(LOGGING_LEVELS["debug"])
            logger.addHandler(file_handler)
            logger.info(f"Logging all LLM Compressor modifier-level logs to {log_path}")

        logger.setLevel(LOGGING_LEVELS["debug"])
        logger.propagate = False

        return logger

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[Union[float, str]],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
        level: Optional[int] = None,
    ) -> bool:
        """
        :param tag: identifying tag to log the values with
        :param value: value to save
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        :param level: level to log at. Corresponds to default logging package levels
        :return: True if logged, False otherwise.
        """
        if not level:
            level = LOGGING_LEVELS["debug"]

        if level > LOGGING_LEVELS["debug"]:
            if step is not None:
                format = "%s %s step %s: %s"
                log_args = [
                    self.name,
                    tag,
                    step,
                    values or value,
                ]
            else:
                format = "%s %s: %s"
                log_args = [self.name, tag, values or value]
        else:
            format = "%s %s [%s - %s]: %s"
            log_args = [self.name, tag, step, wall_time, values or value]

        self._logger.log(level, format, *log_args)

        return True

    def log_string(
        self,
        tag: Optional[str],
        string: Optional[str],
        step: Optional[int],
        wall_time: Optional[float] = None,
        level: Optional[int] = None,
    ) -> bool:
        """
        :param tag: identifying tag to log the values with
        :param string: string to log
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        :param level: level to log at. Corresponds to default logging package levels
        :return: True if logged, False otherwise.
        """
        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(
            tag=tag,
            value=string,
            values=None,
            step=step,
            level=level,
            wall_time=wall_time,
        )


class TensorBoardLogger(LambdaLogger):
    """
    Modifier metrics that handles outputting values into a TensorBoard log directory
    for viewing in TensorBoard.

    :param log_path: the path to create a SummaryWriter at. writer must be None
        to use if not supplied (and writer is None),
        will create a TensorBoard dir in cwd
    :param writer: the writer to log results to,
        if none is given creates a new one at the log_path
    :param name: name given to the metrics, used for identification;
        defaults to tensorboard
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        log_path: str = None,
        writer: SummaryWriter = None,
        name: str = "tensorboard",
        enabled: bool = True,
    ):
        if tensorboard_import_error:
            raise tensorboard_import_error

        if writer and log_path:
            raise ValueError(
                (
                    "log_path given:{} and writer object passed in, "
                    "to create a writer at the log path set writer=None"
                ).format(log_path)
            )
        elif not writer and not log_path:
            log_path = os.path.join("", "tensorboard")

        if os.environ.get("NM_TEST_MODE"):
            test_log_root = os.environ.get("NM_TEST_LOG_DIR")
            log_path = (
                os.path.join(test_log_root, log_path) if log_path else test_log_root
            )

        if log_path:
            _create_dirs(log_path)

        self._writer = writer if writer is not None else SummaryWriter(log_path)
        super().__init__(
            lambda_func=self._log_lambda,
            name=name,
            enabled=enabled,
        )

    @staticmethod
    def available() -> bool:
        """
        :return: True if tensorboard is available and installed, False, otherwise
        """
        return not tensorboard_import_error

    @property
    def writer(self) -> SummaryWriter:
        """
        :return: the writer to log results to,
            if none is given creates a new one at the log_path
        """
        return self._writer

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
        level: Optional[int] = None,
    ) -> bool:
        if value is not None:
            self._writer.add_scalar(tag, value, step, wall_time)

        if values and tag:
            self._writer.add_scalars(tag, values, step, wall_time)
        elif values:
            for name, val in values.items():
                # hyperparameters logging case
                self._writer.add_scalar(name, val, step, wall_time)

        return True


class WANDBLogger(LambdaLogger):
    """
    Modifier metrics that handles outputting values to Weights and Biases.

    :param init_kwargs: the args to call into wandb.init with;
        ex: wandb.init(**init_kwargs). If not supplied, then init will not be called
    :param name: name given to the metrics, used for identification;
        defaults to wandb
    :param enabled: True to log, False otherwise
    """

    @staticmethod
    def available() -> bool:
        """
        :return: True if wandb is available and installed, False, otherwise
        """
        return wandb_available

    def __init__(
        self,
        init_kwargs: Optional[Dict] = None,
        name: str = "wandb",
        enabled: bool = True,
        wandb_err: Optional[Exception] = wandb_err,
    ):
        if wandb_err:
            raise wandb_err

        super().__init__(
            lambda_func=self._log_lambda,
            name=name,
            enabled=enabled,
        )

        if os.environ.get("NM_TEST_MODE"):
            test_log_path = os.environ.get("NM_TEST_LOG_DIR")
            _create_dirs(test_log_path)
            if init_kwargs:
                init_kwargs["dir"] = test_log_path
            else:
                init_kwargs = {"dir": test_log_path}

        if wandb_err:
            raise wandb_err

        if init_kwargs:
            wandb.init(**init_kwargs)
        else:
            wandb.init()

        self.wandb = wandb

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
        level: Optional[int] = None,
    ) -> bool:
        params = {}

        if value:
            params[tag] = value

        if values:
            if tag:
                values = {f"{tag}/{key}": val for key, val in values.items()}
            params.update(values)

        params.update({"Step": step})
        wandb.log(params)

        return True

    def save(
        self,
        file_path: str,
    ) -> bool:
        """
        :param file_path: path to a file to be saved
        """
        wandb.save(file_path)
        return True


class SparsificationGroupLogger(BaseLogger):
    """
    Modifier metrics that handles outputting values to other supported systems.
    Supported ones include:
      - Python logging
      - Tensorboard
      - Weights and Biases
      - Lambda callback

    All are optional and can be bulk disabled and enabled by this root.

    :param lambda_func: an optional lambda function to call back into with any logs.
        The expected call sequence is (tag, value, values, step, wall_time) -> bool
        The return type is True if logged and False otherwise.
    :param python: an optional argument for logging to a python metrics.
        May be a logging.Logger instance to log to, True to create a metrics instance,
        or non truthy to not log anything (False, None)
    :param python_log_level: if python,
        the level to log any incoming data at on the logging.Logger instance
    :param tensorboard: an optional argument for logging to a tensorboard writer.
        May be a SummaryWriter instance to log to, a string representing the directory
        to create a new SummaryWriter to log to, True to create a new SummaryWriter,
        or non truthy to not log anything (False, None)
    :param wandb_: an optional argument for logging to wandb.
        May be a dictionary to pass to the init call for wandb,
        True to log to wandb (will not call init),
        or non truthy to not log anything (False, None)
    :param name: name given to the metrics, used for identification;
        defaults to sparsification
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        lambda_func: Optional[
            Callable[
                [
                    Optional[str],
                    Optional[float],
                    Optional[Dict[str, float]],
                    Optional[int],
                    Optional[float],
                ],
                bool,
            ]
        ] = None,
        python: Optional[Union[bool, Logger]] = None,
        python_log_level: int = logging.INFO,
        tensorboard: Optional[Union[bool, str, SummaryWriter]] = None,
        wandb_: Optional[Union[bool, Dict]] = None,
        name: str = "sparsification",
        enabled: bool = True,
    ):
        super().__init__(name, enabled)
        self._loggers: List[BaseLogger] = []

        if lambda_func:
            self._loggers.append(
                LambdaLogger(lambda_func=lambda_func, name=name, enabled=enabled)
            )

        if python:
            self._loggers.append(
                PythonLogger(
                    logger=python if isinstance(python, Logger) else None,
                    log_level=python_log_level,
                    name=name,
                    enabled=enabled,
                )
            )

        if tensorboard and TensorBoardLogger.available():
            self._loggers.append(
                TensorBoardLogger(
                    log_path=tensorboard if isinstance(tensorboard, str) else None,
                    writer=(
                        tensorboard if isinstance(tensorboard, SummaryWriter) else None
                    ),
                    name=name,
                    enabled=enabled,
                )
            )

        if wandb_ and WANDBLogger.available():
            self._loggers.append(
                WANDBLogger(
                    init_kwargs=wandb_ if isinstance(wandb_, Dict) else None,
                    name=name,
                    enabled=enabled,
                )
            )

    @BaseLogger.enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to log, False otherwise
        """
        self._enabled = value

        for logger in self._loggers:
            logger.enabled = value

    @property
    def loggers(self) -> List[BaseLogger]:
        """
        :return: the created metrics sub instances for this metrics
        """
        return self._loggers

    def log_hyperparams(self, params: Dict, level: Optional[int] = None):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        for logger in self._loggers:
            logger.log_hyperparams(params, level)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        level: Optional[int] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        """
        for logger in self._loggers:
            logger.log_scalar(tag, value, step, wall_time, level)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        level: Optional[int] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        for logger in self._loggers:
            logger.log_scalars(tag, values, step, wall_time, level)


class LoggerManager(ABC):
    """
    Wrapper around loggers that handles log scheduling and handing off logs to intended
    loggers.

    :param loggers: list of loggers assigned to this manager
    :param log_frequency: number of stes or fraction of steps to wait between logs
    :param mode: The logging mode to use, either "on_change" or "exact",
        "on_change" will log when the model has been updated since the last log,
        "exact" will log at the given frequency regardless of model updates.
        Defaults to "exact"
    :param frequency_type: The frequency type to use, either "epoch", "step", or "batch"
        controls what the frequency manager is tracking, e.g. if the frequency type
        is "epoch", then the frequency manager will track the number of epochs that
        have passed since the last log, if the frequency type is "step", then the
        frequency manager will track the number of optimizer steps
    """

    def __init__(
        self,
        loggers: Optional[List[BaseLogger]] = None,
        log_frequency: Union[float, None] = 0.1,
        log_python: bool = True,
        name: str = "manager",
        mode: LoggingModeType = "exact",
        frequency_type: FrequencyType = "epoch",
    ):
        self._name = name
        self._loggers = (
            loggers
            or SparsificationGroupLogger(
                python=log_python,
                name=name,
                tensorboard=True,
                wandb_=True,
            ).loggers
        )

        self.frequency_manager = FrequencyManager(
            mode=mode,
            frequency_type=frequency_type,
            log_frequency=log_frequency,
        )

        self.system = SystemLoggingWraper(
            loggers=self._loggers, frequency_manager=self.frequency_manager
        )
        self.metric = MetricLoggingWrapper(
            loggers=self._loggers, frequency_manager=self.frequency_manager
        )

    def __len__(self):
        return len(self.loggers)

    def __iter__(self):
        return iter(self.loggers)

    def add_logger(self, logger: BaseLogger):
        """
        add a BaseLogger implementation to the loggers of this manager

        :param logger: metrics object to add
        """
        if not isinstance(logger, BaseLogger):
            raise ValueError(f"metrics {type(logger)} must be of type BaseLogger")
        self._loggers.append(logger)

    def log_ready(
        self, current_log_step, last_log_step=None, check_model_update: bool = False
    ):
        """
        Check if there is a metrics that is ready to accept a log

        :param current_log_step: current step log is requested at
        :param last_log_step: last time a log was recorder for this object. (Deprecated)
        :param check_model_update: if True, will check if the model has been updated,
            if False, will only check the log frequency
        :return: True if a metrics is ready to accept a log.
        """
        log_enabled = any(logger.enabled for logger in self.loggers)
        if last_log_step is not None:
            self.frequency_manager.log_written(step=last_log_step)

        return log_enabled and self.frequency_manager.log_ready(
            current_log_step=current_log_step,
            check_model_update=check_model_update,
        )

    def log_written(self, step: LogStepType):
        """
        Update the frequency manager with the last log step written

        :param step: step that was last logged
        """
        self.frequency_manager.log_written(step=step)

    def model_updated(self, step: LogStepType):
        """
        Update the frequency manager with the last model update step

        :param step: step that was last logged
        """
        self.frequency_manager.model_updated(step=step)

    @staticmethod
    def epoch_to_step(epoch, steps_per_epoch):
        return round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    @property
    def loggers(self) -> List[BaseLogger]:
        """
        :return: list of loggers assigned to this manager
        """
        return self._loggers

    @loggers.setter
    def loggers(self, value: List[BaseLogger]):
        """
        :param value: list of loggers assigned to this manager
        """
        self._loggers = value

    @property
    def log_frequency(self) -> Union[str, float, None]:
        """
        :return: number of epochs or fraction of epochs to wait between logs
        """
        return self.frequency_manager._log_frequency

    @log_frequency.setter
    def log_frequency(self, value: Union[str, float, None]):
        """
        :param value: number of epochs or fraction of epochs to wait between logs
        """
        self.frequency_manager._log_frequency = value

    @property
    def name(self) -> str:
        """
        :return: name given to the metrics, used for identification
        """
        return self._name

    @property
    def wandb(self) -> Optional[ModuleType]:
        """
        :return: wandb module if initialized
        """
        for log in self.loggers:
            if isinstance(log, WANDBLogger) and log.enabled:
                return log.wandb
        return None

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        (Note: this method is deprecated and will be removed in a future version,
        use LoggerManager().metric.log_scalar instead)

        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """

        self.metric.log_scalar(
            tag=tag,
            value=value,
            step=step,
            wall_time=wall_time,
            log_types=log_types,
            level=level,
        )

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        (Note: this method is deprecated and will be removed in a future version,
        use LoggerManager().metric.log_scalars instead)

        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """

        self.metric.log_scalars(
            tag=tag,
            values=values,
            step=step,
            wall_time=wall_time,
            log_types=log_types,
            level=level,
        )

    def log_hyperparams(
        self,
        params: Dict,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        (Note: this method is deprecated and will be removed in a future version,
        use LoggerManager().metric.log_hyperparams instead)

        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """

        self.metric.log_hyperparams(
            params=params,
            log_types=log_types,
            level=level,
        )

    def log_string(
        self,
        tag: str,
        string: str,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        (Note: this method is deprecated and will be removed in a future version,
        use LoggerManager().system.log_string instead)

        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        self.system.log_string(
            tag=tag,
            string=string,
            step=step,
            wall_time=wall_time,
            log_types=log_types,
            level=level,
        )

    def save(
        self,
        file_path: str,
        **kwargs,
    ):
        """
        :param file_path: path to a file to be saved
        :param kwargs: additional arguments that a specific metrics might use
        """
        for log in self._loggers:
            if log.enabled:
                log.save(file_path, **kwargs)

    @contextmanager
    def time(self, tag: Optional[str] = None, *args, **kwargs):
        """
        Context manager to log the time it takes to run the block of code

        Usage:
        >>> with LoggerManager().time("my_block"):
        >>>    time.sleep(1)

        :param tag: identifying tag to log the values with
        """

        start = time.time()
        yield
        elapsed = time.time() - start
        if not tag:
            tag = f"{DEFAULT_TAG}_time_secs"
        self.log_scalar(tag=tag, value=float(f"{elapsed:.3f}"), *args, **kwargs)


class LoggingWrapperBase:
    """
    Base class that holds a reference to the loggers and frequency manager
    """

    def __init__(self, loggers: List[BaseLogger], frequency_manager: FrequencyManager):
        self.loggers = loggers
        self._frequency_manager = frequency_manager

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"loggers={self.loggers}, frequency_manager={self._frequency_manager})"
        )


class SystemLoggingWraper(LoggingWrapperBase):
    """
    Wraps utilities and convenience methods for logging strings to the system
    """

    def log_string(
        self,
        tag: str,
        string: str,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        for log in self.loggers:
            if log.enabled and (log_types == ALL_TOKEN or log.name in log_types):
                log.log_string(
                    tag=tag,
                    string=string,
                    step=step,
                    wall_time=wall_time,
                    level=level,
                )

    def debug(self, tag, string, *args, **kwargs):
        """
        logs a string message with level DEBUG on all
        loggers that are enabled

        :param tag: Identifying tag to log the string with
        :param string: The string to log
        :param args: additional arguments to pass to the metrics,
            see `log_string` for more details
        :param kwargs: additional arguments to pass to the metrics,
            see `log_string` for more details
        """
        kwargs["level"] = logging.DEBUG
        self.log_string(tag=tag, string=string, *args, **kwargs)

    def info(self, tag, string, *args, **kwargs):
        """
        logs a string message with level INFO on all
        loggers that are enabled

        :param tag: Identifying tag to log the string with
        :param string: The string to log
        :param args: additional arguments to pass to the metrics,
            see `log_string` for more details
        :param kwargs: additional arguments to pass to the metrics,
            see `log_string` for more details
        """
        kwargs["level"] = logging.INFO
        self.log_string(tag=tag, string=string, *args, **kwargs)

    def warning(self, tag, string, *args, **kwargs):
        """
        logs a string message with level WARNING on all
        loggers that are enabled

        :param tag: Identifying tag to log the string with
        :param string: The string to log
        :param args: additional arguments to pass to the metrics,
            see `log_string` for more details
        :param kwargs: additional arguments to pass to the metrics,
            see `log_string` for more details
        """
        kwargs["level"] = logging.WARNING
        self.log_string(tag=tag, string=string, *args, **kwargs)

    def warn(self, tag, string, *args, **kwargs):
        warnings.warn(
            "The 'warn' method is deprecated, use 'warning' instead",
            DeprecationWarning,
            2,
        )
        self.warning(tag=tag, string=string, *args, **kwargs)

    def error(self, tag, string, *args, **kwargs):
        """
        logs a string message with level ERROR on all
        loggers that are enabled

        :param tag: Identifying tag to log the string with
        :param string: The string to log
        :param args: additional arguments to pass to the metrics,
            see `log_string` for more details
        :param kwargs: additional arguments to pass to the metrics,
            see `log_string` for more details
        """
        kwargs["level"] = logging.ERROR
        self.log_string(tag=tag, string=string, *args, **kwargs)

    def critical(self, tag, string, *args, **kwargs):
        """
        logs a string message with level CRITICAL on all
        loggers that are enabled

        :param tag: Identifying tag to log the string with
        :param string: The string to log
        :param args: additional arguments to pass to the metrics,
            see `log_string` for more details
        :param kwargs: additional arguments to pass to the metrics,
            see `log_string` for more details
        """
        kwargs["level"] = logging.CRITICAL
        self.log_string(tag=tag, string=string, *args, **kwargs)


class MetricLoggingWrapper(LoggingWrapperBase):
    """
    Wraps utilities and convenience methods for logging metrics to the system
    """

    def log_hyperparams(
        self,
        params: Dict,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        for log in self.loggers:
            if log.enabled and (log_types == ALL_TOKEN or log.name in log_types):
                log.log_hyperparams(params, level)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        for log in self.loggers:
            if log.enabled and (log_types == ALL_TOKEN or log.name in log_types):
                log.log_scalar(
                    tag=tag,
                    value=value,
                    step=step,
                    wall_time=wall_time,
                    level=level,
                )

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        level: Optional[int] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        :param kwargs: additional logging arguments to support Python and custom loggers
        :return: True if logged, False otherwise.
        """
        for log in self.loggers:
            if log.enabled and (log_types == ALL_TOKEN or log.name in log_types):
                log.log_scalars(
                    tag=tag,
                    values=values,
                    step=step,
                    wall_time=wall_time,
                    level=level,
                )

    def add_scalar(
        self,
        value,
        tag: str = DEFAULT_TAG,
        step: Optional[int] = None,
        wall_time: Union[int, float, None] = None,
        **kwargs,
    ):
        """
        Add a scalar value to the metrics

        :param value: value to log
        :param tag: tag to log the value with, defaults to DEFAULT_TAG
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :param kwargs: additional logging arguments to to pass through to the
            metrics
        """
        self.log_scalar(tag=tag, value=value, step=step, wall_time=wall_time, **kwargs)

    def add_scalars(
        self,
        values: Dict[str, Any],
        tag: str = DEFAULT_TAG,
        step: Optional[int] = None,
        wall_time: Union[int, float, None] = None,
        **kwargs,
    ):
        """
        Adds multiple scalar values to the metrics

        :param values: values to log, must be A dict of serializable
            python objects i.e `str`, `ints`, `floats`, `Tensors`, `dicts`, etc
        :param tag: tag to log the value with, defaults to DEFAULT_TAG
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :param kwargs: additional logging arguments to to pass through to the
            metrics
        """
        self.log_scalars(
            tag=tag, values=values, step=step, wall_time=wall_time, **kwargs
        )

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        tag: Optional[str] = DEFAULT_TAG,
        **kwargs,
    ) -> None:
        """
        :param data:  A dict of serializable python objects i.e `str`,
                `ints`, `floats`, `Tensors`, `dicts`, etc
        :param step: global step for when the values were taken
        :param tag: identifying tag to log the values with, defaults to DEFAULT_TAG
        :param kwargs: additional logging arguments to support
            Python and custom loggers
        """
        self.log_scalars(tag=tag, values=data, step=step, **kwargs)


def _create_dirs(path: str):
    path = Path(path).expanduser().absolute()
    path.mkdir(parents=True, exist_ok=True)
