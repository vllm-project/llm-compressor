"""
Module for managing LLM Compressor state.

Provides classes for holding and updating the state information
related to data, hardware, and model compression.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from llmcompressor.metrics import BaseLogger, LoggerManager

__all__ = ["State", "Data", "Hardware", "ModifiedState"]


@dataclass
class Data:
    """
    A dataclass to hold different data sets for training, validation,
    testing, and/or calibration. Each data set is a ModifiableData instance.

    :param train: The training data set
    :type train: Optional[Any]
    :param val: The validation data set
    :type val: Optional[Any]
    :param test: The testing data set
    :type test: Optional[Any]
    :param calib: The calibration data set
    :type calib: Optional[Any]
    """

    train: Optional[Any] = None
    val: Optional[Any] = None
    test: Optional[Any] = None
    calib: Optional[Any] = None


@dataclass
class Hardware:
    """
    A dataclass to hold information about the hardware being used.

    :param device: The current device being used for training
    :type device: Optional[str]
    :param devices: List of all devices to be used for training
    :type devices: Optional[List[str]]
    :param rank: The rank of the current device
    :type rank: Optional[int]
    :param world_size: The total number of devices being used
    :type world_size: Optional[int]
    :param local_rank: The local rank of the current device
    :type local_rank: Optional[int]
    :param local_world_size: The total number of devices being used on the local machine
    :type local_world_size: Optional[int]
    :param distributed: Whether or not distributed training is being used
    :type distributed: Optional[bool]
    :param distributed_strategy: The distributed strategy being used
    :type distributed_strategy: Optional[str]
    """

    device: Optional[str] = None
    devices: Optional[List[str]] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None
    local_world_size: Optional[int] = None
    distributed: Optional[bool] = None
    distributed_strategy: Optional[str] = None


@dataclass
class State:
    """
    State class holds information about the current compression state.

    :param model: The model being used for compression
    :type model: Any
    :param teacher_model: The teacher model being used for compression
    :type teacher_model: Any
    :param optimizer: The optimizer being used for training
    :type optimizer: Any
    :param optim_wrapped: Whether or not the optimizer has been wrapped
    :type optim_wrapped: bool
    :param loss: The loss function being used for training
    :type loss: Any
    :param batch_data: The current batch of data being used for compression
    :type batch_data: Any
    :param data: The data sets being used for training, validation, testing,
        and/or calibration, wrapped in a Data instance
    :type data: Data
    :param hardware: Hardware instance holding info about the target hardware being used
    :type hardware: Hardware
    :param loggers: LoggerManager instance holding all the loggers to log
    :type loggers: Optional[LoggerManager]
    :param model_log_cadence: The cadence to log model information w.r.t epochs.
        If 1, logs every epoch. If 2, logs every other epoch, etc. Default is 1.
    :type model_log_cadence: Optional[float]
    """

    model: Any = None
    teacher_model: Any = None
    optimizer: Any = None
    optim_wrapped: bool = None
    loss: Any = None
    batch_data: Any = None
    data: Data = field(default_factory=Data)
    hardware: Hardware = field(default_factory=Hardware)
    loggers: Optional[LoggerManager] = None
    model_log_cadence: Optional[float] = None
    _last_log_step: Union[float, int, None] = None

    @property
    def compression_ready(self) -> bool:
        """
        Check if the model and optimizer are set for compression.

        :return: True if model and optimizer are set, False otherwise
        :rtype: bool
        """
        ready = self.model is not None and self.optimizer is not None
        logger.debug("Compression ready: {}", ready)
        return ready

    def update(
        self,
        model: Any = None,
        teacher_model: Any = None,
        optimizer: Any = None,
        attach_optim_callbacks: bool = True,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        calib_data: Any = None,
        copy_data: bool = True,
        start: float = None,
        steps_per_epoch: int = None,
        batches_per_step: int = None,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
        model_log_cadence: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        Update the state with the given parameters.

        :param model: The model to update the state with
        :type model: Any
        :param teacher_model: The teacher model to update the state with
        :type teacher_model: Any
        :param optimizer: The optimizer to update the state with
        :type optimizer: Any
        :param attach_optim_callbacks: Whether or not to attach optimizer callbacks
        :type attach_optim_callbacks: bool
        :param train_data: The training data to update the state with
        :type train_data: Any
        :param val_data: The validation data to update the state with
        :type val_data: Any
        :param test_data: The testing data to update the state with
        :type test_data: Any
        :param calib_data: The calibration data to update the state with
        :type calib_data: Any
        :param copy_data: Whether or not to copy the data
        :type copy_data: bool
        :param start: The start index to update the state with
        :type start: float
        :param steps_per_epoch: The steps per epoch to update the state with
        :type steps_per_epoch: int
        :param batches_per_step: The batches per step to update the state with
        :type batches_per_step: int
        :param loggers: The metrics manager to setup logging important info and
            milestones to, also accepts a list of BaseLogger(s)
        :type loggers: Union[None, LoggerManager, List[BaseLogger]]
        :param model_log_cadence: The cadence to log model information w.r.t epochs.
            If 1, logs every epoch. If 2, logs every other epoch, etc. Default is 1.
        :type model_log_cadence: Optional[float]
        :param kwargs: Additional keyword arguments to update the state with
        :return: The updated state as a dictionary
        :rtype: Dict
        """
        logger.debug(
            "Updating state with provided parameters: {}",
            {
                "model": model,
                "teacher_model": teacher_model,
                "optimizer": optimizer,
                "attach_optim_callbacks": attach_optim_callbacks,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "calib_data": calib_data,
                "copy_data": copy_data,
                "start": start,
                "steps_per_epoch": steps_per_epoch,
                "batches_per_step": batches_per_step,
                "loggers": loggers,
                "model_log_cadence": model_log_cadence,
                "kwargs": kwargs,
            },
        )

        if model is not None:
            self.model = model
        if teacher_model is not None:
            self.teacher_model = teacher_model
        if optimizer is not None:
            self.optim_wrapped = attach_optim_callbacks
            self.optimizer = optimizer
        if train_data is not None:
            self.data.train = train_data if not copy_data else deepcopy(train_data)
        if val_data is not None:
            self.data.val = val_data if not copy_data else deepcopy(val_data)
        if test_data is not None:
            self.data.test = test_data if not copy_data else deepcopy(test_data)
        if calib_data is not None:
            self.data.calib = calib_data if not copy_data else deepcopy(calib_data)

        if "device" in kwargs:
            self.hardware.device = kwargs["device"]

        loggers = loggers or []
        if isinstance(loggers, list):
            loggers = LoggerManager(loggers)
        self.loggers = loggers

        if model_log_cadence is not None:
            self.model_log_cadence = model_log_cadence

        return kwargs


@dataclass
class ModifiedState:
    """
    A dataclass to represent a modified model, optimizer, and loss.

    :param model: The modified model
    :type model: Optional[Any]
    :param optimizer: The modified optimizer
    :type optimizer: Optional[Any]
    :param loss: The modified loss
    :type loss: Optional[Any]
    :param modifier_data: The modifier data used to modify the
        model, optimizer, and loss
    :type modifier_data: Optional[List[Dict[str, Any]]]
    """

    model: Optional[Any] = None
    optimizer: Optional[Any] = None
    loss: Optional[Any] = None
    modifier_data: Optional[List[Dict[str, Any]]] = None

    def __init__(self, model, optimizer, loss, modifier_data):
        """
        Initialize the ModifiedState with the given parameters.

        :param model: The modified model
        :type model: Any
        :param optimizer: The modified optimizer
        :type optimizer: Any
        :param loss: The modified loss
        :type loss: Any
        :param modifier_data: The modifier data used to modify the model, optimizer,
            and loss
        :type modifier_data: List[Dict[str, Any]]
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.modifier_data = modifier_data
