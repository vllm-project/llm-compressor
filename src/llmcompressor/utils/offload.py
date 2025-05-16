import torch
from accelerate.utils import has_offloaded_params

__all__ = ["has_device_parameters", "has_device_execution_hook", "has_device_execution"]


def has_device_parameters(model: torch.nn.Module) -> bool:
    return any(
        param.device not in (torch.device("cpu"), torch.device("meta"))
        for param in model.parameters()
    )


def has_device_execution_hook(model: torch.nn.Module) -> bool:
    return any(
        has_offloaded_params(module) and module._hf_hook.execution_device != "cpu"
        for module in model.modules()
    )


def has_device_execution(model: torch.nn.Module) -> bool:
    return has_device_execution_hook(model) or has_device_parameters(model)
