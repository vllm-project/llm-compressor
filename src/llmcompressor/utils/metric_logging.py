"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking GPU memory usage, measuring model
layer sizes, and comprehensive logging during compression workflows.
Supports both NVIDIA and AMD GPU monitoring with detailed memory
statistics and performance metrics.
"""

import os
import time
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from loguru import logger
from torch.nn import Module

__all__ = ["CompressionLogger"]

GPUMemory = namedtuple("GPUMemory", ["id", "pct_used", "total"])


class GPUType(Enum):
    nv = "nv"
    amd = "amd"


def get_layer_size_mb(module: Module) -> float:
    param_size = 0
    buffer_size = 0

    for param in module.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size
    total_size_mb = total_size / (1e6)  # Convert bytes to MB

    return total_size_mb


class CompressionLogger:
    """
    Log metrics related to compression algorithm

    :param start_tick: time when algorithm started
    :param losses: loss as result of algorithm
    :param gpu_type: device manufacturer (e.g. Nvidia, AMD)
    :param visible_ids: list of device ids visible to current process
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None
        self.loss = None
        self.gpu_type = GPUType.amd if torch.version.hip else GPUType.nv

        # Parse appropriate env var for visible devices to monitor
        # If env var is unset, default to all devices
        self.visible_ids = []
        visible_devices_env_var = (
            "CUDA_VISIBLE_DEVICES"
            if self.gpu_type == GPUType.nv
            else "AMD_VISIBLE_DEVICES"
        )
        visible_devices_str = os.environ.get(visible_devices_env_var, "")
        try:
            self.visible_ids = list(
                map(
                    int,
                    visible_devices_str.lstrip("[").rstrip("]").split(","),
                )
            )
        except Exception:
            logger.bind(log_once=True).warning(
                f"Could not parse {visible_devices_env_var}. "
                "All devices will be monitored"
            )

    def set_loss(self, loss: float):
        self.loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()
        patch = logger.patch(lambda r: r.update(function="compress"))

        if self.start_tick is not None:
            duration = stop_tick - self.start_tick
            patch.log("METRIC", f"time {duration:.2f}s")
        if self.loss is not None:
            patch.log("METRIC", f"error {self.loss:.2f}")

        gpu_usage: List[GPUMemory] = self.get_GPU_memory_usage()
        for gpu in gpu_usage:
            perc = gpu.pct_used * 100
            patch.log(
                "METRIC",
                (
                    f"GPU {gpu.id} | usage: {perc:.2f}%"
                    f" | total memory: {gpu.total:.1f} GB"
                ),
            )

        compressed_size = get_layer_size_mb(self.module)
        patch.log("METRIC", f"Compressed module size: {compressed_size} MB")

    def get_GPU_memory_usage(self) -> List[GPUMemory]:
        if self.gpu_type == GPUType.amd:
            return self._get_GPU_usage_amd(self.visible_ids)
        else:
            return self._get_GPU_usage_nv(self.visible_ids)

    @staticmethod
    def _get_GPU_usage_nv(visible_ids: List[int]) -> List[GPUMemory]:
        """
        get gpu usage for visible Nvidia GPUs using nvml lib

        :param visible_ids: list of GPUs to monitor.
            If unset or zero length, defaults to all
        """
        try:
            import pynvml
            from pynvml import NVMLError

            try:
                pynvml.nvmlInit()
            except NVMLError as _err:
                logger.warning(f"Pynml library error:\n {_err}")
                return []

            usage: List[GPUMemory] = []

            if len(visible_ids) == 0:
                visible_ids = range(pynvml.nvmlDeviceGetCount())

            for id in visible_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage_percentage = mem_info.used / mem_info.total
                total_memory_gb = mem_info.total / (1e9)
                usage.append(GPUMemory(id, memory_usage_percentage, total_memory_gb))
            pynvml.nvmlShutdown()
            return usage

        except ImportError:
            logger.warning("Failed to obtain GPU usage from pynvml")
            return []

    @staticmethod
    def _get_GPU_usage_amd(visible_ids: List[int]) -> List[GPUMemory]:
        """
        get gpu usage for AMD GPUs using amdsmi lib

        :param visible_ids: list of GPUs to monitor.
            If unset or zero length, defaults to all
        """
        usage: List[GPUMemory] = []
        try:
            import amdsmi

            try:
                amdsmi.amdsmi_init()
                devices = amdsmi.amdsmi_get_processor_handles()

                if len(visible_ids) == 0:
                    visible_ids = range(len(devices))

                for id in visible_ids:
                    device = devices[id]
                    vram_memory_usage = amdsmi.amdsmi_get_gpu_memory_usage(
                        device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                    )
                    vram_memory_total = amdsmi.amdsmi_get_gpu_memory_total(
                        device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                    )

                    memory_percentage = vram_memory_usage / vram_memory_total
                    usage.append(
                        GPUMemory(id, memory_percentage, vram_memory_total / (1e9)),
                    )
                amdsmi.amdsmi_shut_down()
            except amdsmi.AmdSmiException as error:
                logger.warning(f"amdsmi library error:\n {error}")
        except ImportError:
            logger.warning("Failed to obtain GPU usage from amdsmi")

        return usage
