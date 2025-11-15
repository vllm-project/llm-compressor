from typing import Any

import torch
from contextlib import nullcontext, contextmanager

from .cache import DeviceCache
from .utils import send_to_device

_offloaded_module_subclasses: dict[str, type] = dict()


class OffloadMixin(torch.nn.Module):
    _direct_attributes = {
        # core attributes
        "__class__",
        "__dict__",
        "__weakref__",

        # instance attributes
        "_module",
        "_cache",
        "_disable_offloading",
        "disable_offloading",
        "disable_onloading",

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
        self._modules = module.__dict__["_modules"]
        # TODO: account for buffers

    def __getattribute__(self, name: str) -> object:
        if name in OffloadMixin._direct_attributes:
            return object.__getattribute__(self, name)

        elif name in self._module._parameters:
            value = self._module._parameters[name]
            if value is not None:
                return self._cache[value]
            else:
                return None

        else:
            return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any):
        if name in OffloadMixin._direct_attributes:
            return object.__setattr__(self, name, value)
        
        elif name in self._module._parameters:
            old_value = self._module._parameters[name]
            if old_value is not None:
                self._cache[old_value] = value

        setattr(self._module, name, value)

    def __delattr__(self, name: str):
        if name in OffloadMixin._direct_attributes:
            return object.__delattr__(self, name)

        elif name in self._module._parameters:
            old_value = self._module._parameters[name]
            if old_value is not None:
                del self._cache[old_value]

        delattr(self._module, name)

    def __call__(self, *args, **kwargs):
        # note that *args unpacking is not traceable
        # (even though the unpacking is purely virtual here)
        # some approaches to consider:
        # 1. try to get torch.fx to ignore this part, maybe using is_tracing conditionals?
        # 2. make __getattribute__ redirect to the original method if _disable_offloading == Fals
        # 3. only `offload_module` wrap modules with parameters (assume not seq ancestors)
        args, kwargs = send_to_device(args, self._cache.onload_device), send_to_device(kwargs, self._cache.onload_device)

        if self._disable_offloading:
            with self._cache.disable_offloading():
                return self._module.__call__.__func__(self, *args, **kwargs)
        else:
            return self._module.__call__.__func__(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        args, kwargs = send_to_device(args, self._cache.onload_device), send_to_device(kwargs, self._cache.onload_device)

        if self._disable_offloading:
            with self._cache.disable_offloading():
                return self._module.forward.__func__(self, *args, **kwargs)
        else:
            return self._module.forward.__func__(self, *args, **kwargs)

    @contextmanager
    def disable_offloading(self):
        with self._cache.disable_offloading():
            yield

    @contextmanager
    def disable_onloading(self):
        with self._cache.disable_onloading():
            yield

    def execution_device(self) -> torch.device | str:
        return self._cache.onload_device
        
        

# might want this to be a method of DeviceCache
# depending on plans for disk offloading
def offload_module(
    module: torch.nn.Module,
    cache: DeviceCache,
    disable_offloading: bool = False
) -> OffloadMixin:
    class_name = module.__class__.__name__
    if class_name not in _offloaded_module_subclasses:
        _offloaded_module_subclasses[class_name] = make_offload_module_subclass(module.__class__)

    return _offloaded_module_subclasses[class_name](module, cache, disable_offloading)

def make_offload_module_subclass(parent_cls: type) -> type:
    subclass = type(f"Offloaded{parent_cls.__name__}", (OffloadMixin, parent_cls), {})
    subclass.__name__ = parent_cls.__name__

    assert issubclass(subclass, parent_cls)
    return subclass