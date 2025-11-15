from collections.abc import Mapping

import torch

import warnings


def clone_to_device(value: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    if isinstance(value, torch.nn.Parameter):
        if value.requires_grad:
            # warn once:
            warnings.warn("Torch Offloader does not currently support parameters with gradients")
        return torch.nn.Parameter(value.to(device), value.requires_grad)
    else:
        return value.to(device)


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return isinstance(data, tuple) and hasattr(data, "_asdict") and hasattr(data, "_fields")

def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    if isinstance(tensor, torch.Tensor) or hasattr(tensor, "to"):
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
        
    elif isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor)
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for k, t in tensor.items()
            }
        )
    else:
        return tensor