"""Offload orchestration for pure MoE linearization and repacking."""

import torch
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
    subgraph_unload_modules,
)

from .linear_experts import LinearExperts2D
from .linearize import (
    get_moe_modules,
    linearize_moe,
    repack_moe,
)


def _named_modules(name: str, module: torch.nn.Module):
    return {
        name if not relative_name else f"{name}.{relative_name}": child
        for relative_name, child in module.named_modules()
    }


def _target_modules(
    model: torch.nn.Module,
    subgraph_modules: dict[str, torch.nn.Module] | None,
    linearized: bool,
):
    module_set = (
        {
            submodule
            for module in subgraph_modules.values()
            for submodule in module.modules()
        }
        if subgraph_modules is not None
        else set(model.modules())
    )
    moe_lookup = get_moe_modules(model)
    return [
        (moe_lookup[module], module)
        for module in module_set
        if module in moe_lookup
        and isinstance(module, LinearExperts2D) == linearized
    ]


def _collect_modules(entries):
    modules = {}
    for name, module in entries:
        modules.update(_named_modules(name, module))
    return modules


def _source_policy(name: str, offload_kwargs: dict[str, dict]):
    if name in offload_kwargs:
        return offload_kwargs[name]
    return next(
        (
            kwargs
            for child_name, kwargs in offload_kwargs.items()
            if child_name.startswith(f"{name}.")
        ),
        None,
    )


def _set_replacement_policy(
    name: str,
    replacement: torch.nn.Module,
    policy: dict | None,
    offload_kwargs: dict[str, dict],
) -> tuple[dict[str, torch.nn.Module], dict[str, dict]] | None:
    for child_name in list(offload_kwargs):
        if child_name == name or child_name.startswith(f"{name}."):
            del offload_kwargs[child_name]

    if policy is None:
        return None

    replacement_modules = _named_modules(name, replacement)
    replacement_kwargs = {
        child_name: policy for child_name in replacement_modules
    }
    offload_kwargs.update(replacement_kwargs)
    return replacement_modules, replacement_kwargs


def linearize_moe_with_offload(
    model: torch.nn.Module,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    cpu_materialize: bool = True,
    onload_replacements: bool = False,
):
    """Materialize, linearize, and optionally onload MoE replacements."""
    entries = _target_modules(model, subgraph_modules, linearized=False)
    operation_modules = (
        subgraph_modules if subgraph_modules is not None else _collect_modules(entries)
    )
    source_modules = _collect_modules(entries)
    execution_devices = {
        name: get_execution_device(module) for name, module in entries
    }

    if cpu_materialize:
        offload_kwargs = subgraph_unload_modules(source_modules)
    else:
        offload_kwargs = subgraph_onload_modules(source_modules)
    policies = {
        name: _source_policy(name, offload_kwargs) for name, _ in entries
    }

    linearize_moe(model, operation_modules)

    for name, _ in entries:
        replacement = model.get_submodule(name)
        policy = policies[name]
        result = _set_replacement_policy(
            name, replacement, policy, offload_kwargs
        )
        if result is None:
            replacement.to(device=execution_devices[name])
            continue

        replacement_modules, replacement_kwargs = result
        subgraph_offload_modules(replacement_modules, replacement_kwargs)
        if onload_replacements:
            subgraph_onload_modules(replacement_modules)

    return offload_kwargs


def repack_moe_with_offload(
    model: torch.nn.Module,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    offload_kwargs: dict[str, dict] | None = None,
):
    """Onload, repack, and re-offload linearized MoE replacements."""
    entries = _target_modules(model, subgraph_modules, linearized=True)
    operation_modules = (
        subgraph_modules if subgraph_modules is not None else _collect_modules(entries)
    )
    source_modules = _collect_modules(entries)
    if offload_kwargs is None:
        offload_kwargs = subgraph_onload_modules(source_modules)
    policies = {
        name: _source_policy(name, offload_kwargs) for name, _ in entries
    }

    repack_moe(model, operation_modules)

    for name, _ in entries:
        replacement = model.get_submodule(name)
        result = _set_replacement_policy(
            name, replacement, policies[name], offload_kwargs
        )
        if result is not None:
            replacement_modules, replacement_kwargs = result
            subgraph_offload_modules(replacement_modules, replacement_kwargs)

    return offload_kwargs
