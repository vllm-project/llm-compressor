import contextlib
from functools import wraps
from typing import Any, Callable, ClassVar, Optional, Set, Union

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

    Typical example
    >>> modifier.register_forward_hook(module, hook)
    >>> with HooksMixin.disable_hooks():
            model.forward(...)
    >>> modifier.remove_hooks()

    Example of activating only a specific subset of hooks
    >>> hooks = [modifier.register_forward_hook(module, hook) for module in ...]
    >>> with HooksMixin.disable_hooks(keep=hooks):
            model.forward(...)
    >>> modifier.remove_hooks(hooks)
    """

    # attached to global HooksMixin class
    _HOOKS_DISABLED: ClassVar[bool] = False
    _HOOKS_KEEP_ENABLED: ClassVar[Set[RemovableHandle]] = set()

    # attached to local subclasses
    _hooks: Set[RemovableHandle] = set()

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls, keep: Set[RemovableHandle] = frozenset()):
        """
        Disable all hooks across all modifiers. Composing multiple contexts is
        equivalent to the union of `keep` arguments

        :param keep: optional set of handles to keep enabled
        """
        try:
            cls._HOOKS_DISABLED = True
            cls._HOOKS_KEEP_ENABLED |= keep
            yield
        finally:
            cls._HOOKS_DISABLED = False
            cls._HOOKS_KEEP_ENABLED -= keep

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
        handle = None

        @wraps(hook)
        def wrapped_hook(*args, **kwargs):
            nonlocal handle

            if (
                HooksMixin._HOOKS_DISABLED
                and handle not in HooksMixin._HOOKS_KEEP_ENABLED
            ):
                return

            return hook(*args, **kwargs)

        register_function = getattr(target, f"register_{hook_type}_hook")
        handle = register_function(wrapped_hook, **kwargs)
        self._hooks.add(handle)
        logger.debug(f"{self} added {handle}")

        return handle

    def remove_hooks(self, handles: Optional[Set[RemovableHandle]] = None):
        """
        Removes hooks registered by this modifier

        :param handles: optional list of handles to remove, defaults to all hooks
            registerd by this modifier
        """
        if handles is None:
            handles = self._hooks

        for hook in handles:
            hook.remove()

        self._hooks -= handles
