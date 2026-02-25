"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking device memory usage, loss, and runtime
during module compression (optimization). Supports both NVIDIA and AMD GPU monitoring
"""

import time

import torch
from loguru import logger

__all__ = ["CompressionLogger"]


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

    def set_loss(self, loss: float):
        self.loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()
        patch = logger.patch(lambda r: r.update(function="compress"))

        if self.start_tick is not None:
            patch.log("METRIC", f"time {(stop_tick - self.start_tick):.2f}s")
        if self.loss is not None:
            patch.log("METRIC", f"error {self.loss:.2f}")

        for device_id in range(torch.cuda.device_count()):
            max_memory = torch.cuda.max_memory_allocated(device_id)
            used_memory = torch.cuda.get_device_properties(device_id).total_memory
            perc_used = 100 * used_memory / max_memory
            patch.log(
                "METRIC",
                (
                    f"GPU {device_id} | usage: {perc_used:.2f}%"
                    f" | total memory: {max_memory:.1f} GB"
                ),
            )
