import gc

import torch
from compressed_tensors.utils import deprecated

__all__ = ["measure_accelerator_memory", "measure_cuda_memory"]


class measure_accelerator_memory:
    def __init__(self, device=None):
        self.device = device

    def reset_peak_memory_stats(self):
        torch.accelerator.reset_peak_memory_stats(self.device)

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        self.reset_peak_memory_stats()
        mem = torch.accelerator.max_memory_allocated(self.device)
        return mem

    def peak_memory_usage(self) -> float:
        # Return the peak memory usage in bytes since the last reset
        mem = torch.accelerator.max_memory_allocated(self.device)
        return mem

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.overall_peak_memory = self.peak_memory_usage()
        self.peak_consumed_memory = self.overall_peak_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


@deprecated("measure_accelerator_memory")
def measure_cuda_memory(device=None):
    return measure_accelerator_memory(device=device)
