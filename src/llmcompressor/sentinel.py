"""
Sentinel value implementation for LLM compression workflows.

Provides unique sentinel values following PEP 661 with Pydantic validation
support. Used for creating distinct marker objects that can be safely compared
by identity across the compression framework.
"""

import inspect

from pydantic_core import core_schema

_registry = {}


class Sentinel:
    """
    Unique sentinel values. Implements https://peps.python.org/pep-0661/
    with dummy pydantic validation
    """

    def __new__(cls, name, module_name=None):
        name = str(name)

        if module_name is None:
            module_name = inspect.currentframe().f_globals.get("__file__")
            if module_name is None:
                module_name = __name__

        registry_key = f"{module_name}-{name}"

        sentinel = _registry.get(registry_key, None)
        if sentinel is not None:
            return sentinel

        sentinel = super().__new__(cls)
        sentinel._name = name
        sentinel._module_name = module_name

        return _registry.setdefault(registry_key, sentinel)

    def __repr__(self):
        return self._name

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._name,
                self._module_name,
            ),
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v)
            ),
        )

    @classmethod
    def validate(cls, value: "Sentinel") -> "Sentinel":
        return value
