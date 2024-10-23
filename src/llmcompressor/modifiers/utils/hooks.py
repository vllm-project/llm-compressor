import contextlib

import torch

__all__ = ["HooksMixin"]


class HooksMixin:
    """"
    Class to manage the registration, disabling, and removal of hooks. Registering
    and removing hooks should be handled by modifier classes which inherit from this
    mixin, while disabling hooks should disable all hooks across modifiers.

    Modifiers which implement hooks should use the @HooksMixin.hook decorator
    Modifiers must pass registered hooks handles to self.register_hook() and must
    remove hooks when finished using self.remove_hooks()
    """
    HOOKS_DISABLED: bool = False

    @classmethod
    def hook(cls, func):
        def wrapped(*args, **kwargs):
            if cls.HOOKS_DISABLED:
                return

            func(*args, **kwargs)

        return wrapped

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls):
        """
        Disable all hooks across all modifiers
        TODO: select which modifier hooks are disabled/ kept enabled
        """
        try:
            cls.HOOKS_DISABLED = True
            yield
        finally:
            cls.HOOKS_DISABLED = False

    def __init__(self):
        self._hooks = []

    def register_hook(self, handle: torch.utils.hooks.RemovableHandle):
        """
        Usage: self.register_hook(module.register_forward_hook(...))

        :param handle: handle of added hook
        """
        self._hooks.append(handle)

    def remove_hooks(self):
        """
        Remove all hooks belonging to a modifier
        """
        for hook in self._hooks:
            hook.remove()
