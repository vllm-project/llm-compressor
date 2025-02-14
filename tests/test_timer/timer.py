import time
from collections import defaultdict
from contextlib import contextmanager
from threading import RLock

import numpy
from loguru import logger

__all__ = ["Timer"]


class Timer:
    """Timer to log timings during inference requests. Should be used through
    timer_utils.get_singleton_manager()
    """

    _instance = None

    def __init__(self, enable_logging: bool):
        """
        :param enable_logging: whether or not time logging is enabled
        """
        self.enable_logging = enable_logging
        self.measurements = defaultdict(float)
        self.lock = RLock()

    @contextmanager
    def time(self, func_name: str):
        """
        Time the given function, add the time to the dictionary of lists
        tracking each function and calculate the average, if
        self.avg_after_iterations number of iterations have passed.

        :param func_name: name of the function that will be used to log the
            measurements
        """
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            with self.lock:
                self._update(func_name, end - start)

    def _update(self, func_name: str, time_elapased: float):
        """
        Update the dictionary of measurements for each function call

        :param func_name: name of the function that will be used to log the
            measurements
        :param time_elapsed: time taken for function execution
        """
        if self.measurements.get(func_name) is None:
            self.measurements[func_name] = [time_elapased]
        else:
            self.measurements[func_name].append(time_elapased)

    def _compute_average(self):
        """
        Display averages time for each logged function. After the average is
        calculated, clear the measurements.

        """
        with self.lock:
            for func_name in self.measurements:
                logger.info(f"Average time for {func_name}: ")
                logger.info(str(numpy.average(self.measurements.get(func_name))))
                self.measurements[func_name].clear()
