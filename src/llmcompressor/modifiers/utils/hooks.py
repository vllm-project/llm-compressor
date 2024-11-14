import contextlib
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Set, Tuple, Union

import torch
from loguru import logger
from pydantic import BaseModel
from torch.utils.hooks import RemovableHandle
from collections import defaultdict

from llmcompressor.modifiers.utils.pytorch_helpers import EarlyStopException
from llmcompressor.utils.helpers import getattr_chain
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.pytorch.module import get_layers, get_no_split_params

__all__ = ["HooksMixin", "LayerCompressorMixin"]


class HooksMixin(BaseModel):
    """
    Class to manage the registration, disabling, and removal of hooks. Registering
    and removing hooks should be handled by modifier classes which inherit from this
    mixin, while disabling hooks should disable all hooks across modifiers.

    Modifiers which implement hooks should register them using
    `self.register_..._hook(module, hook)` rather than the usual
    `module.register_..._hook(hook)`. Modifiers should remove hooks with
    `self.remove_hooks()`

    Lifecycle:
        - self = Modifier(HooksMixin)(...)
        - self.register_forward_hook(module, hook)
        - with HooksMixin.disable_hooks(): model.forward()
        - self.remove_hooks()
    """

    _HOOKS_DISABLED: ClassVar[bool] = False     # attached to global HooksMixin
    _hooks: List[RemovableHandle] = []          # attached to local subclasses

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls):
        """
        Disable all hooks across all modifiers
        TODO: select which modifier hooks are disabled/ kept enabled
        """
        try:
            cls._HOOKS_DISABLED = True
            yield
        finally:
            cls._HOOKS_DISABLED = False

    def register_forward_pre_hook(
        self,
        module: torch.nn.Module,
        func: Callable[[Any], Any],
        **kwargs,
    ):
        self._register_hook("register_forward_pre_hook", module, func, **kwargs)

    def register_forward_hook(
        self,
        module: torch.nn.Module,
        func: Callable[[Any], Any],
        **kwargs,
    ):
        self._register_hook("register_forward_hook", module, func, **kwargs)

    def remove_hooks(self):
        """
        Remove all hooks belonging to a modifier
        """
        for hook in self._hooks:
            hook.remove()

    def _register_hook(
        self,
        register_func_name: str,
        module: torch.nn.Module,
        func: Callable[[Any], Any],
        **kwargs,
    ):
        @wraps(func)
        def wrapped_hook(*args, **kwargs):
            if HooksMixin._HOOKS_DISABLED:
                return
            
            return func(*args, **kwargs)
        
        handle = getattr(module, register_func_name)(wrapped_hook, **kwargs)
        self._hooks.append(handle)