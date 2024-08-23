from typing import List, Tuple

from loguru import logger
from torch.nn import Module


def get_GPU_memory_usage() -> List[Tuple]:
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
            total_memory_gb = mem_info.total / (1024**3)
            usage.append(
                (memory_usage_percentage, total_memory_gb),
            )
        pynvml.nvmlShutdown()
        return usage

    except ImportError:
        logger.warning("Failed to obtain GPU usage from pynvml")
        return []


def get_layer_size_bytes(module: Module) -> float:
    param_size = 0
    buffer_size = 0

    for param in module.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size
    total_size_mb = total_size / (1024**2)  # Convert bytes to MB

    return total_size_mb
