"""
Built-in architecture support modules for MoNE.
"""

import importlib

_BUILTIN_MONE_SUPPORT_MODULES = (
    "llmcompressor.modeling.moe.minimax_mone",
)


def load_builtin_mone_support() -> None:
    for module_name in _BUILTIN_MONE_SUPPORT_MODULES:
        importlib.import_module(module_name)
