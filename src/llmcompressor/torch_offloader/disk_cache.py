from weakref import WeakValueDictionary

import os
import torch
import contextlib
import tempfile
import uuid

from safetensors.torch import load_file, save_file, safe_open


class DiskCache:
    """
    The disk cache handles the onloading of tensors from disk to an onload device.

    When used with module offloading,
    assumes that the model starts on the meta device
    TODO: need to write a script to load models while creating an offload_files mapping

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor.
    """
    def __init__(self, offload_files: dict[torch.Tensor, str], device: torch.device | str):
        self.onload_device = device

        # flags for disabling
        self.onloading_disabled: bool = False
        self.offloading_disabled: bool = False

        # offloaded files
        self.offload_files: dict[torch.Tensor, str] = offload_files
        self.update_files: dict[torch.Tensor, str] = {}

        # this caching layer controls for shared tensors
        self.onload_values: WeakValueDictionary[torch.Tensor, torch.Tensor] = WeakValueDictionary()  # offloaded (meta) tensors -> onloaded tensors

        # cache onloaded values when offloading is disabled
        self.keep_onloaded_values: set[torch.Tensor] = set()

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        if self.onloading_disabled:
            return key
        
        # onload value, potentially from cache
        if key not in self.onload_values:

            # onload value from disk
            if key in self.update_files:
                file_path = self.update_files[key]
            elif key in self.offload_files:
                file_path = self.offload_files[key]
            else:
                raise ValueError()
            with safe_open(file_path, framework="pt", device=self.onload_device) as file:
                value = file.get_tensor(hash(key))

            self.onload_values[key] = value

        else:
            value = self.onload_values[key]

        # if offloading is disabled, keep a strong reference (to keep the value alive)
        if self.offloading_disabled:
            self.keep_onloaded_values.add(value)

        return value

    def __delitem__(self, key: torch.Tensor):
        # remove any strong references to onloaded values
        # won't actually delete from disk. Future gets for the same key will succeed,
        # but this is nbd and is guarded upstream by OffloadedModule._offload_names
        if (
            self.offloading_disabled
            and key in self.onload_values
            and self.onload_values[key] in self.keep_onloaded_values
        ):
            self.keep_onloaded_values.remove(self.onload_values[key])

    def __setitem__(self, key: torch.Tensor, value: torch.Tensor | None):
        # create new diff files when updating
        if key not in self.update_files:
            self.update_files[key] = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        if value is not None:
            tensors = load_file(self.update_files[key])
            tensors[hash(key)] = value
            save_file(tensors, self.update_files[key])

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
