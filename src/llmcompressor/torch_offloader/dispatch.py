
from collections.abc import Container

import torch

from .cache import DeviceCache
from .module import offload_module

import contextlib


def dispatch_model(
    model: torch.nn.Module,
    device: torch.device | str,
    no_split_modules: Container[str] = tuple(),
):
    # each model needs to reference a shared cache because
    # we have to coordinate the onloading of shared tensors
    cache = DeviceCache(device)

    memo = set()
    offloaded = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if module in memo:
            continue

        disable_offloading = module.__class__.__name__ in no_split_modules
        offloaded_module = offload_module(module, cache, disable_offloading)
        offloaded.add(offloaded_module)

        if name != "":
            model.set_submodule(name, offloaded_module)
        else:
            model = offloaded_module

        memo.add(module)

    assert offloaded == set(model.modules())

    @contextlib.contextmanager
    def disable_offloading():
        with cache.disable_offloading():
            yield

    model.disable_offloading = disable_offloading

    return model


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)