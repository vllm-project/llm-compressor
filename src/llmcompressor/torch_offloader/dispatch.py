from collections.abc import Container

import warnings
import torch
from contextlib import contextmanager

from .cache import DeviceCache
from .module import offload_module, OffloadMixin


def dispatch_model(
    model: torch.nn.Module,
    device: torch.device | str,
    no_split_modules: Container[str] = tuple(),
) -> torch.nn.Module:
    # each model needs to reference a shared cache because
    # we have to coordinate the onloading of shared tensors
    cache = DeviceCache(device)

    memo = set()
    offloaded = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if module in memo:
            continue

        disable_offloading = module.__class__.__name__ in no_split_modules
        if len(module._parameters) > 0:
            offloaded_module = offload_module(module, cache, disable_offloading)
        else:
            offloaded_module = module
        offloaded.add(offloaded_module)

        if name != "":
            model.set_submodule(name, offloaded_module)
        else:
            model = offloaded_module

        memo.add(module)

    assert offloaded == set(model.modules())

    @contextmanager
    def disable_offloading():
        with cache.disable_offloading():
            yield

    @contextmanager
    def disable_onloading():
        with cache.disable_onloading():
            yield

    def execution_device() -> torch.device | str:
        return cache.onload_device
    
    def device(self) -> torch.device:
        return torch.device(cache.onload_device)  # maybe cast to device
    
    model.disable_offloading = disable_offloading
    model.disable_onloading = disable_onloading
    model.execution_device = execution_device

    # TODO: get this to work to ignore annnoying generation messages
    # might have to patch parent class (ew)
    #del model.device
    #model.device = device.__get__(model)

    return model


def update_offload_parameter(module: torch.nn.Module, name: str, data: torch.Tensor):
    if isinstance(module, OffloadMixin) or hasattr(module, "disable_onloading"):
        with module.disable_onloading():
            getattr(module, name).copy_(data)
    else:
        getattr(module, name).copy_(data)


def get_execution_device(module: torch.nn.Module) -> torch.device | str:
    if isinstance(module, OffloadMixin) or hasattr(module, "execution_device"):
        return module.execution_device()

    else:
        first_param = next(module.parameters(), None)
        if first_param is not None:
            return first_param.device
            
        warnings.warn(f"Unable to get execution device of {module}, falling back to CPU")
        return torch.device("cpu")
