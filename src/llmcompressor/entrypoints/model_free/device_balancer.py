import functools
import inspect
import multiprocessing.context
from typing import List, Optional, Union
import multiprocessing

import torch
from loguru import logger

from llmcompressor.entrypoints.model_free.helpers import gpu_if_available

__all__ = ["DeviceLoadBalancer"]


class DeviceLoadBalancer:
    """
    Load balancer for distributing jobs across multiple GPU devices.
    Tracks device usage and provides the least busy device when requested.
    """

    def __init__(
        self,
        device: Optional[
            Union[torch.device, str, List[Union[torch.device, str]]]
        ],
        mp_context: multiprocessing.context.BaseContext,
    ):
        """
        Initialize the load balancer with device(s).

        :param device: Device specification - can be:
            - None: auto-select available GPU (cuda, xpu, npu) or fallback to CPU
            - Single device: torch.device or str (e.g., "cuda:0")
            - List of devices: List[torch.device | str] for multi-GPU support
        """
        # Parse device argument into list of devices
        if isinstance(device, list):
            # Multi-GPU: validate and convert each device
            device_list = [gpu_if_available(d) for d in device]
        else:
            # Single device: create list with single device
            device_list = [gpu_if_available(device)]

        manager = mp_context.Manager()
        self.devices: list[str | int] = device_list
        self.device_usage = manager.dict({device: 0 for device in self.devices})
        self.lock = manager.Lock()

    def get_device(self) -> torch.device:
        """
        Get the least busy device. Thread-safe.

        :return: The device with the fewest active jobs
        """
        with self.lock:
            # Find device with minimum usage
            device = min(self.device_usage.keys(), key=lambda d: self.device_usage[d])
            self.device_usage[device] += 1
            return device

    def release_device(self, device: torch.device):
        """
        Release a device back to the pool. Thread-safe.

        :param device: The device to release
        """
        with self.lock:
            if device in self.device_usage:
                self.device_usage[device] -= 1
            else:
                logger.warning(f"Attempted to release unknown device: {device}")

    @staticmethod
    def inject_device(func):
        """
        Decorator that manages device lifecycle for functions.

        The decorated function should have a 'device' parameter. When calling
        the wrapped function, pass a DeviceLoadBalancer instance in place of
        the device parameter. The decorator will automatically:
        1. Get a device from the load balancer
        2. Call the function with that device
        3. Release the device when complete (even if an exception occurs)

        :param func: Function to decorate (must have a 'device' parameter)
        :return: Wrapped function that accepts load_balancer instead of device
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = dict(bound_args.arguments)

            load_balancer: DeviceLoadBalancer = kwargs.pop("device")
            device = load_balancer.get_device()
            kwargs["device"] = device

            try:
                return func(**kwargs)
            finally:
                load_balancer.release_device(device)

        return wrapper
