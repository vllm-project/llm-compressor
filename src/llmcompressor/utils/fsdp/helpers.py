import operator
from typing import Optional

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:
    FullyShardedDataParallel = None

from torch.nn import Module

from llmcompressor.core.state import State

__all__ = [
    "is_fsdp_model",
    "maybe_get_wrapped",
    "set_wrapped_model",
    "get_fsdp_parent",
]


def is_fsdp_model(model: Module) -> bool:
    """
    Check if a model instance is wrapped by FSDP

    :param model: pytorch model to check
    :return: True if module is wrapped, False otherwise
    """
    if not FullyShardedDataParallel:
        return False

    return isinstance(model, FullyShardedDataParallel)


def maybe_get_wrapped(model: Module) -> Module:
    """
    Given a model that may or may not have a distributed wrapper, return the underlying
    wrapped model.

    :param model: input model to get wrapped model from
    :returns: wrapped model
    """
    if is_fsdp_model(model=model):
        return model._fsdp_wrapped_module
    return model


def set_wrapped_model(state: State, wrapped_model: Module):
    """
    Given a state with a model that may or may not have a distributed wrapper, set
    the underlying wrapped model.

    :param state: state to update model of
    :param updated_wrapped: model to inject into input_model
    """
    if is_fsdp_model(state.model):
        state.model._fsdp_wrapped_module = wrapped_model
    else:
        state.model = wrapped_model


def get_fsdp_parent(layer_name: str, model: Module) -> Optional[Module]:
    """
    Gets the closest parent of layer_name that is wrapped by FSDP. If no FSDP wrapper
    is found just return None

    :param layer_name: layer name in model to get parent of
    :model: pytorch module to search through
    :return: FSDP wrapped parent of layer_name if available, otherwise None
    """
    if not is_fsdp_model(model):
        return None

    parent_name = layer_name
    parent = operator.attrgetter(parent_name)(model)
    while not isinstance(parent, FullyShardedDataParallel):
        if len(parent_name) == 0:  # we've reached the root module and its not FSDP
            # this should never get hit because we check for an FSDP root above
            # but while statements without a backup are too scary
            return None
        parent_name = ".".join(parent_name.split(".")[:-1])
        parent = operator.attrgetter(parent_name)(model)

    return parent
