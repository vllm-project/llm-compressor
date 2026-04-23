"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking device memory usage, loss, and runtime
during module compression (optimization). Supports both NVIDIA and AMD GPU monitoring
"""

import time
from typing import Iterable, Optional

import torch
from compressed_tensors.offload import is_distributed
from loguru import logger

__all__ = ["CompressionLogger"]


class CompressionLogger:
    """
    Log metrics related to compression algorithms
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None

        self._name = None
        self._loss = None

    def set_results(
        self,
        name: Optional[str] = None,
        loss: Optional[float] = None,
    ):
        self._name = name
        self._loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()

        patch = logger.patch(lambda r: r.update(function=(self._name or "compress")))

        patch.log("METRIC", f"time {(stop_tick - self.start_tick):.2f}s")
        if self._loss is not None:
            patch.log("METRIC", f"error {self._loss:.2f}")

        if not torch.accelerator.is_available():
            return

        for device_id in _get_visible_devices():
            used_memory = torch.accelerator.max_memory_allocated(device_id) / 1e9
            max_memory = torch.accelerator.get_memory_info(device_id)[1] / 1e9
            perc_used = 100 * used_memory / max_memory
            patch.log(
                "METRIC",
                (
                    f"Accelerator {device_id} | usage: {perc_used:.2f}%"
                    f" | total memory: {max_memory:.1f} Gb"
                ),
            )


def _get_visible_devices() -> Iterable:
    if is_distributed():
        return [torch.accelerator.current_device_index()]

    else:
        return range(torch.accelerator.device_count())
