import gc

import torch
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


def onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """
    Onload a list of modules, returning the kwargs used for offloading.
    """
    offload_kwargs = {}
    for name, module in modules.items():
        if isinstance(module._parameters, OffloadCache):
            init_kwargs = get_cache_init_kwargs(module)
            offload_kwargs[name] = init_kwargs
            set_onload_device(module, "cpu_pin")
            remove_module_offload(module, onload_tensors=True)
    return offload_kwargs


def offload_modules(
    modules: dict[str, torch.nn.Module],
    offload_kwargs: dict[str, dict],
):
    """
    Offload a list of modules, using the provided device map and kwargs.
    """
    for name, module in modules.items():
        if name in offload_kwargs:
            offload_module(module, **offload_kwargs[name])
