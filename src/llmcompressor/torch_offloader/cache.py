from typing import Any

from weakref import WeakValueDictionary

import torch
import contextlib

from .utils import clone_to_device


class DeviceCache:
    """
    The device cache handles the onloading of tensors from an offload device to
    an onload device.

    When used with module offloading,
    assumes that the model starts on the offload device

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor.
    """
    def __init__(self, device: torch.device | str):
        self.onload_device = device

        # flags for disabling
        self.onloading_disabled: bool = False
        self.offloading_disabled: bool = False

        # onloaded values cache
        self.onload_values: WeakValueDictionary[torch.Tensor, torch.Tensor] = WeakValueDictionary()  # offloaded tensors -> onloaded tensors

        # strong ref to values to disable offloading
        self.keep_onloaded_values: set[torch.Tensor] = set()

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        if self.onloading_disabled:
            return key

        # onload value, potentially from cache
        if key not in self.onload_values:

            # onload value from (cpu)
            onloaded_value = clone_to_device(key, self.onload_device)
            self.onload_values[key] = onloaded_value

        else:
            onloaded_value = self.onload_values[key]

        # if offloading is disabled, keep a strong reference (to keep the value alive)
        if self.offloading_disabled:
            self.keep_onloaded_values.add(onloaded_value)

        return onloaded_value

    def __delitem__(self, key: torch.Tensor):
        # remove any strong references to onloaded values
        if (
            self.offloading_disabled
            and key in self.onload_values
            and self.onload_values[key] in self.keep_onloaded_values
        ):
            self.keep_onloaded_values.remove(self.onload_values[key])

    def __setitem__(self, key: torch.Tensor, value: Any):
        # invaldiate onloaded values cache
        del self[key]

    @contextlib.contextmanager
    def disable_offloading(self):
        if self.offloading_disabled:
            raise ValueError("Already in `disable_offloading` context!")

        self.offloading_disabled = True
        self.keep_onloaded_values.update(self.onload_values.values())

        yield

        self.offloading_disabled = False
        self.keep_onloaded_values.clear()

    @contextlib.contextmanager
    def disable_onloading(self):
        if self.onloading_disabled:
            raise ValueError("Already in `disable_onloading` context!")
        
        self.onloading_disabled = True

        yield

        self.onloading_disabled = False
