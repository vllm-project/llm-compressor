import gc

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


def onload(
    modules: dict[str, torch.nn.Module],
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Onload a list of modules, returning the device map and kwargs used for offloading
    `device_map` and `kwargs` are keyed by the name, not the actual module itself.
    This simplifies swapping out modules during linearization and repacking, as the
    names remain consistent.
    """
    device_map = {}
    kwargs = {}
    for name, module in modules.items():
        if isinstance(module._parameters, OffloadCache):
            init_kwargs = get_cache_init_kwargs(module)
            device_map[name] = init_kwargs["onload_device"]
            kwargs[name] = init_kwargs
            remove_module_offload(module, onload_tensors=True)
    return device_map, kwargs


def offload(
    modules: dict[str, torch.nn.Module],
    device_map: dict[str, str],
    kwargs: dict[str, dict],
):
    """
    Offload a list of modules, using the provided device map and kwargs.
    """
    for name, module in modules.items():
        if name in device_map:
            offload_module(module, **kwargs[name])

    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()


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
