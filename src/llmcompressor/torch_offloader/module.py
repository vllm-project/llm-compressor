from typing import Any, TypeVar

import torch
from contextlib import nullcontext

from .cache import DeviceCache
from .utils import send_to_device

_offloaded_module_subclasses: dict[str, type] = dict()

T = TypeVar("", bound=torch.nn.Module)

# might want this to be a method of DeviceCache
# depending on plans for disk offloading
def offload_module(
    module: T,
    cache: DeviceCache,
    disable_offloading: bool = False
) -> T:
    if len(module._parameters) <= 0:
        return module


    class_name = module.__class__.__name__
    if class_name not in _offloaded_module_subclasses:
        _offloaded_module_subclasses[class_name] = make_offload_module_subclass(module.__class__)

    return _offloaded_module_subclasses[class_name](module, cache, disable_offloading)



def make_offload_module_subclass(parent_cls: type) -> type:
    class OffloadedModule(parent_cls):
        __name__ = parent_cls.__name__
        _direct_attributes = {
            # core attributes
            "__class__",
            "__dict__",
            "__weakref__",

            # instance attributes
            "_module",
            "_cache",
            "_disable_offloading",
            "_offload_names",

            # these functions will return the wrapped `_module` unless we call with self
            "modules",
            "named_modules",
            "_modules",

            # call path
            "__call__",
            "_compiled_call_impl",
            "_call_impl",
            "forward",
        }

        def __init__(self, module: torch.nn.Module, cache: DeviceCache, disable_offloading: bool):
            self._module = module
            self._cache = cache
            self._disable_offloading = disable_offloading
            self._offload_names = set(module.__dict__["_parameters"].keys())
            self._modules = module.__dict__["_modules"]

        def __getattribute__(self, name: str) -> object:
            if name in OffloadedModule._direct_attributes:
                return object.__getattribute__(self, name)

            elif name in self._offload_names:
                value = self._module._parameters[name]

                if value is not None:
                    return self._cache[value]
                else:
                    return None

            else:
                return self._module.__getattribute__(name)

        def __setattr__(self, name: str, value: Any):
            if name in OffloadedModule._direct_attributes:
                return object.__setattr__(self, name, value)
            
            elif name in self._offload_names:
                old_value = self._module._parameters[name]

                if old_value is not None:
                    self._cache[old_value] = value

            self._module.__setattr__(name, value)

        def __delattr__(self, name: str):
            if name in OffloadedModule._direct_attributes:
                return object.__delattr__(self, name)

            elif name in self._offload_names:
                old_value = self._module._parameters[name]

                if old_value is not None:
                    del self._cache[old_value]
                    
                self._offload_names.remove(name)

            self._module.__delattr__(name)

        def __call__(self, *args, **kwargs):
            args, kwargs = send_to_device(args, self._cache.onload_device), send_to_device(kwargs, self._cache.onload_device)

            with self._cache.disable_offloading() if self._disable_offloading else nullcontext():
                return self._module.__call__.__func__(self, *args, **kwargs)

        def forward(self, *args, **kwargs):
            args, kwargs = send_to_device(args, self._cache.onload_device), send_to_device(kwargs, self._cache.onload_device)

            with self._cache.disable_offloading() if self._disable_offloading else nullcontext():
                return self._module.forward.__func__(self, *args, **kwargs)

    return OffloadedModule