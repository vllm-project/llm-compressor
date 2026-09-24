import warnings

import torch
from compressed_tensors.offload import get_cache_init_kwargs
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import (  # noqa: F401
    dispatch_model,
    dispatch_with_map,
    get_device_map,
    offload_model,
    remove_dispatch,
)
from compressed_tensors.offload.module import (
    offload_module,
    remove_module_offload,
    stage_module_offload,
)


def stage_modules(
    modules: dict[str, torch.nn.Module],
    pin_memory: bool = False,
) -> None:
    """Stage offloaded module tensors in CPU memory for a later onload."""
    for name, module in modules.items():
        if not isinstance(module._parameters, OffloadCache):
            warnings.warn(f"Module {name} is not offloaded. Skipping staging.")
            continue

        stage_module_offload(module, pin_memory=pin_memory)


def onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """Onload modules, optionally consuming tensors staged in CPU memory."""
    offload_kwargs = {}
    for name, module in modules.items():
        if isinstance(module._parameters, OffloadCache):
            init_kwargs = get_cache_init_kwargs(module)
            offload_kwargs[name] = init_kwargs

            remove_module_offload(module, onload_tensors=True)
        else:
            warnings.warn(f"Module {name} is not offloaded. Skipping onload.")
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
        else:
            warnings.warn(f"No offload kwargs provided for module {name}. Using defaults.")
            offload_kwargs = get_cache_init_kwargs(module)
            offload_module(module)
