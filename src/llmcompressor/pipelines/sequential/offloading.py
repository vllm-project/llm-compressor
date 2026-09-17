import contextlib
import gc
from collections.abc import Iterable

import torch
from compressed_tensors.logger import logger
from compressed_tensors.offload import get_cache_init_kwargs
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import (  # noqa: F401
    dispatch_model,
    dispatch_with_map,
    get_device_map,
    offload_model,
    remove_dispatch,
    set_onload_device,
)
from compressed_tensors.offload.module import (
    offload_module,
    remove_module_offload,
)

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D


class OnloadWrapper:
    """
    Class for wrapping onloaded modules. This is the only way for
    the disable_offloading_controlled context manager to work, since
    it needs to keep track of modules after linearization/packing,
    which creates new module instances.
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ):
        self.module = module
        self.module._onload_wrapper = self

        # Must be captured while the module is still offloaded: after onloading,
        # get_offloaded_device would return the execution device instead
        self.offloading_info = (
            get_cache_init_kwargs(module)
            if isinstance(module._parameters, OffloadCache)
            else None
        )

    def onload(self):
        """
        Onload the module's parameters and buffers to the specified onload device.
        """
        if isinstance(self.module, LinearExperts2D):
            for m in self.module.modules():
                remove_module_offload(m, onload_tensors=True)
        else:
            remove_module_offload(self.module, onload_tensors=True)

    def offload(self):
        """
        Offload the module's parameters and buffers to the specified offload device.
        """
        if isinstance(self.module, LinearExperts2D):
            for m in self.module.modules():
                # Nested modules can have their own wrapper in the active context.
                # Skip those here so each module is offloaded exactly once.
                if (
                    m is not self.module
                    and hasattr(m, "_onload_wrapper")
                    and m._onload_wrapper is not self
                ):
                    continue
                if self.offloading_info is not None:
                    offload_module(m, **self.offloading_info)
        else:
            if self.offloading_info is not None:
                offload_module(self.module, **self.offloading_info)

        del self.module._onload_wrapper

    def replace_with(self, new_module: torch.nn.Module):
        """
        Replace the wrapped module with a new module.
        This is useful for linearization/packing,
        which creates new module instances.
        """
        self.module = new_module
        self.module._onload_wrapper = self


def _log_cuda_memory(prefix: str):
    """
    Emit a lightweight CUDA memory snapshot for debugging.
    """
    if not torch.accelerator.is_available():
        logger.debug(f"{prefix} | CUDA unavailable")
        return

    device = torch.accelerator.current_device_index()
    allocated = torch.accelerator.memory_allocated(device) / 1024**3
    reserved = torch.accelerator.memory_reserved(device) / 1024**3
    peak_allocated = torch.accelerator.max_memory_allocated(device) / 1024**3
    logger.debug(
        f"{prefix} | cuda:{device} allocated={allocated:.2f} GiB "
        f"reserved={reserved:.2f} GiB peak_allocated={peak_allocated:.2f} GiB"
    )


@contextlib.contextmanager
def disable_offloading_controlled(
    model: torch.nn.Module,
    subgraph: Iterable[torch.nn.Module] | None = None,
):
    """
    Context manager to disable offloading for a specific subgraph of the model.

    Intended for sequential pipeline. Onload the entire subgraph, then offload it.

    :param model: the full model
    :param subgraph: iterable of modules in the subgraph to onload
    """
    # deduplicate in case the subgraph contains nested modules, since each unique
    # module can only be offloaded once
    modules = subgraph if subgraph is not None else model.modules()
    modules_list = list(dict.fromkeys(modules))
    wrapper_list = [OnloadWrapper(module) for module in modules_list]

    try:
        _log_cuda_memory("disable_offloading_controlled: before onload")
        for wrapper in wrapper_list:
            if wrapper.offloading_info is None:
                continue
            wrapper.onload()

        _log_cuda_memory("disable_offloading_controlled: after onload")
        yield

    finally:
        _log_cuda_memory("disable_offloading_controlled: before offload")
        for wrapper in wrapper_list:
            if wrapper.offloading_info is None:
                continue
            wrapper.offload()

        # This is pretty much required since we don't create enough objects
        # to trigger the garbage collector to run on its own,
        # and we want to free memory as soon as possible
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        _log_cuda_memory("disable_offloading_controlled: after offload")
