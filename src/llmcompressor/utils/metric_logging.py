import time
from typing import List, Tuple

import torch
from loguru import logger
from torch.nn import Module

__all__ = ["get_GPU_memory_usage", "get_layer_size_mb", "CompressionLogger"]


def get_GPU_memory_usage() -> List[Tuple[float, float]]:
    if torch.version.hip:
        return get_GPU_usage_amd()
    else:
        return get_GPU_usage_nv()


def get_GPU_usage_nv() -> List[Tuple[float, float]]:
    """
    get gpu usage for Nvidia GPUs using nvml lib
    """
    try:
        import pynvml
        from pynvml import NVMLError

        try:
            pynvml.nvmlInit()
        except NVMLError as _err:
            logger.warning(f"Pynml library error:\n {_err}")
            return []

        device_count = pynvml.nvmlDeviceGetCount()
        usage = []  # [(percentage, total_memory_MB)]

        # Iterate through all GPUs
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage_percentage = mem_info.used / mem_info.total
            total_memory_gb = mem_info.total / (1e9)
            usage.append(
                (memory_usage_percentage, total_memory_gb),
            )
        pynvml.nvmlShutdown()
        return usage

    except ImportError:
        logger.warning("Failed to obtain GPU usage from pynvml")
        return []


def get_GPU_usage_amd() -> List[Tuple[float, float]]:
    """
    get gpu usage for AMD GPUs using amdsmi lib
    """
    usage = []
    try:
        import amdsmi

        try:
            amdsmi.amdsmi_init()
            devices = amdsmi.amdsmi_get_processor_handles()

            for device in devices:
                vram_memory_usage = amdsmi.amdsmi_get_gpu_memory_usage(
                    device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                )
                vram_memory_total = amdsmi.amdsmi_get_gpu_memory_total(
                    device, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM
                )

                memory_percentage = vram_memory_usage / vram_memory_total
                usage.append(
                    (memory_percentage, vram_memory_total / (1e9)),
                )
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiException as error:
            logger.warning(f"amdsmi library error:\n {error}")
    except ImportError:
        logger.warning("Failed to obtain GPU usage from amdsmi")

    return usage


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

    :param start_tick: time when algorithm started"
    :param losses: loss as result of algorithm
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None
        self.loss = None

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

        gpu_usage = get_GPU_memory_usage()
        if len(gpu_usage) > 0:
            for i in range(len(gpu_usage)):
                perc = gpu_usage[i][0] * 100
                total_memory = int(gpu_usage[i][1])  # GB
                patch.log(
                    "METRIC",
                    (
                        f"GPU {i} | usage: {perc:.2f}%"
                        f" | total memory: {total_memory} GB"
                    ),
                )

        compressed_size = get_layer_size_mb(self.module)
        patch.log("METRIC", f"Compressed module size: {compressed_size} MB")
