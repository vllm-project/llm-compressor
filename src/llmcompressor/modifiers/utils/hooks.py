import contextlib
from functools import wraps
from typing import Any, Callable, ClassVar, List, Union

import torch
from loguru import logger
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
    `self.remove_hooks()`.
    Hooks can be applied to modules or parameters
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
        target: Union[torch.nn.Module, torch.nn.Parameter],
        hook: Callable[[Any], Any],
        hook_type: str,
        **kwargs,
    ) -> RemovableHandle:
        """
        Registers a hook on a specified module/parameter with the option to disable it
        with HooksMixin.disable_hooks()
        :param target: the module or parameter on which the hook should be registered
        :param hook: the hook to register
        :param hook_type: the type of hook to register corresponding to the
            `register_{hook_type}_hook` attribute on torch.nn.Module.
            Ex. "forward", "forward_pre", "full_backward", "state_dict_post", ""
        :param kwargs: keyword arguments to pass to register hook method
        """
        @wraps(hook)
        def wrapped_hook(*args, **kwargs):
            if HooksMixin._HOOKS_DISABLED:
                return

            return hook(*args, **kwargs)

        register_function = getattr(target, f"register_{hook_type}_hook")
        handle = register_function(wrapped_hook, **kwargs)
        self._hooks.append(handle)
        logger.debug(f"{self} added {handle}")

        return handle

    def remove_hooks(self):
        """Remove all hooks belonging to a modifier"""
        for hook in self._hooks:
            hook.remove()

        self._hooks = []