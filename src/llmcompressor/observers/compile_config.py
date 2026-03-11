"""
Global configuration for observer torch.compile support.

The compile flag is set by the oneshot entrypoint and read by observer
instances at call time. This avoids threading the flag through recipe
and modifier layers.
"""

_enable_observer_compile: bool = False


def set_observer_compile(enabled: bool) -> None:
    global _enable_observer_compile
    _enable_observer_compile = enabled


def get_observer_compile() -> bool:
    return _enable_observer_compile
