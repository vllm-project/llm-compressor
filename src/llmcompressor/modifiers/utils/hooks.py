import contextlib
from functools import wraps
from typing import Any, Callable, ClassVar, List

import torch
from pydantic import BaseModel
from torch.utils.hooks import RemovableHandle

__all__ = ["HooksMixin"]


class HooksMixin(BaseModel):
    """
    Mixin to manage hook registration, disabling, and removal.
    Modifiers should use `self.register_hook(module, hook, hook_type)`
    for hook registration and `self.remove_hooks()` for removal.

    Modifiers which implement hooks should register them using
    `self.register_..._hook(module, hook)` rather than the usual
    `module.register_..._hook(hook)`. Modifiers should remove hooks with
    `self.remove_hooks()`

    Lifecycle:
        - modifier.register_forward_hook(module, hook)
        - with HooksMixin.disable_hooks(): model.forward()
        - modifier.remove_hooks()
    """

    _HOOKS_DISABLED: ClassVar[bool] = False  # attached to global HooksMixin
    _hooks: List[RemovableHandle] = []  # attached to local subclasses

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls):
        """Disable all hooks across all modifiers"""
        try:
            cls._HOOKS_DISABLED = True
            yield
        finally:
            cls._HOOKS_DISABLED = False

    def register_hook(
        self,
        module: torch.nn.Module,
        func: Callable[[Any], Any],
        hook_type: str,
        **kwargs,
    ):
        @wraps(func)
        def wrapped_hook(*args, **kwargs):
            if HooksMixin._HOOKS_DISABLED:
                return

            return func(*args, **kwargs)

        handle = getattr(module, f"register_{hook_type}_hook")(wrapped_hook, **kwargs)
        self._hooks.append(handle)

    def remove_hooks(self):
        """
        Remove all hooks belonging to a modifier
        """
        for hook in self._hooks:
            hook.remove()
