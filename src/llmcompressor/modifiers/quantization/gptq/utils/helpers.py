import time
from typing import Any, Iterable, List, Tuple, Union

import torch
from loguru import logger

from llmcompressor.utils.metric_logging import (
    get_GPU_memory_usage,
    get_layer_size_bytes,
)

__all__ = ["get_output_error", "gptq_hook", "MetricsLogger"]


def get_output_error(
    uncompressed: Tuple[torch.Tensor, ...],
    compressed: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Calculate mean absolute error between weight-uncompressed outputs and
    weight-compressed outputs

    :param uncompressed: uncompressed-weight outputs
    :param compressed: compressed-weight outputs
    :return: mean absolute error between outputs
    """
    # assume first output is the the relevant output (true for most Modules)
    uncompressed = uncompressed[0]
    compressed = compressed[0]

    return torch.mean(torch.abs(uncompressed - compressed))
    
def gptq_hook(func):
    def wrapped(self, *args, **kwargs):
        if self._hooks_disabled:
            return

        func(self, *args, **kwargs)

    return wrapped


class MetricsLogger:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None
        self.losses = None

    def set_losses(self, losses: torch.Tensor):
        self.losses = losses

    def __enter__(self) -> "MetricsLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Log metrics related to compression algorithm

        :param start_tick: time when algorithm started"
        :param losses: loss as result of algorithm
        """
        patch = logger.patch(lambda r: r.update(function="compress"))

        if self.start_tick is not None:
            patch.log("METRIC", "time %.2f" % (time.time() - self.start_tick))
        if self.losses is not None:
            patch.log("METRIC", "error %.2f" % torch.sum(self.losses).item())

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

        compressed_size = get_layer_size_bytes(self.module)
        patch.log("METRIC", f"Compressed layer size: {compressed_size} MB")
