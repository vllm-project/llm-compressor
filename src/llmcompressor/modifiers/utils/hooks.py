import contextlib

import torch

__all__ = ["HooksMixin"]


class HooksMixin:
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
        try:
            cls.HOOKS_DISABLED = True
            yield
        finally:
            cls.HOOKS_DISABLED = False

    def __init__(self):
        self._hooks = []

    def register_hook(self, handle: torch.utils.hooks.RemovableHandle):
        self._hooks.append(handle)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
