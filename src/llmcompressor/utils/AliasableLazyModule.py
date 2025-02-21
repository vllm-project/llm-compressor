from typing import Any, Dict, Tuple

from transformers.utils import _LazyModule

__all__ = ["_AliasableLazyModule"]


class _AliasableLazyModule(_LazyModule):
    """
    Extends _LazyModule to support aliases names

    >>> _file = globals()["__file__"]
    >>> sys.modules["animals"] = _AliasableLazyModule(
        name="animals,
        module_file=_file,
        import_structure=define_import_structure(_file),
        module_spec=__spec__,
        aliases={
            "PigWithLipstick": ("mammals", "Pig"),
        }
    >>> from animals import PigWithLipstick
    """

    def __init__(self, *args, aliases: Dict[str, Tuple[str, str]], **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = aliases

    def __getattr__(self, name: str) -> Any:
        if name in self._aliases:
            module_name, name = self._aliases[name]
            module = self._get_module(module_name)
            return getattr(module, name)

        return super().__getattr__(name)
