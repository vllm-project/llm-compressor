# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Universal registry to support registration and loading of child classes and plugins
of neuralmagic utilities
"""

import importlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union


__all__ = [
    "RegistryMixin",
    "register",
    "get_from_registry",
    "registered_names",
    "registered_aliases",
    "standardize_lookup_name",
]


_ALIAS_REGISTRY: Dict[Type, Dict[str, str]] = defaultdict(dict)
_REGISTRY: Dict[Type, Dict[str, Any]] = defaultdict(dict)


def standardize_lookup_name(name: str) -> str:
    """
    Standardize the given name for lookup in the registry.
    This will replace all underscores and spaces with hyphens and
    convert the name to lowercase.

    example:
    ```
    standardize_lookup_name("Foo_bar baz") == "foo-bar-baz"
    ```

    :param name: name to standardize
    :return: standardized name
    """
    return name.replace("_", "-").replace(" ", "-").lower()


def standardize_alias_name(
    name: Union[None, str, List[str]]
) -> Union[None, str, List[str]]:
    if name is None:
        return None
    elif isinstance(name, str):
        return standardize_lookup_name(name)
    else:  # isinstance(name, list)
        return [standardize_lookup_name(n) for n in name]


class RegistryMixin:
    """
    Universal registry to support registration and loading of child classes and plugins
    of neuralmagic utilities.

    Classes that require a registry or plugins may add the `RegistryMixin` and use
    `register` and `load` as the main entrypoints for adding new implementations and
    loading requested values from its registry.

    If a class should only have its child classes in its registry, the class should
    set the static attribute `registry_requires_subclass` to True

    example
    ```python
    class Dataset(RegistryMixin):
        pass


    # register with default name
    @Dataset.register()
    class ImageNetDataset(Dataset):
        pass

    # load as "ImageNetDataset"
    imagenet = Dataset.load("ImageNetDataset")

    # register with custom name
    @Dataset.register(name="cifar-dataset")
    class Cifar(Dataset):
        pass

    Note: the name will be standardized for lookup in the registry.
    For example, if a class is registered as "cifar_dataset" or
    "cifar dataset", it will be stored as "cifar-dataset". The user
    will be able to load the class with any of the three name variants.

    # register with multiple aliases
    @Dataset.register(alias=["cifar-10-dataset", "cifar_100_dataset"])
    class Cifar(Dataset):
        pass

    # load as "cifar-dataset"
    cifar = Dataset.load_from_registry("cifar-dataset")

    # load from custom file that implements a dataset
    mnist = Dataset.load_from_registry("/path/to/mnnist_dataset.py:MnistDataset")
    ```
    """

    # set to True in child class to add check that registered/retrieved values
    # implement the class it is registered to
    registry_requires_subclass: bool = False

    @classmethod
    def register(
        cls, name: Optional[str] = None, alias: Union[List[str], str, None] = None
    ):
        """
        Decorator for registering a value (ie class or function) wrapped by this
        decorator to the base class (class that .register is called from)

        :param name: name or list of names to register the wrapped value as,
            defaults to value.__name__
        :param alias: alias or list of aliases to register the wrapped value as,
            defaults to None
        :return: register decorator
        """

        def decorator(value: Any):
            cls.register_value(value, name=name, alias=alias)
            return value

        return decorator

    @classmethod
    def register_value(
        cls, value: Any, name: str, alias: Union[str, List[str], None] = None
    ):
        """
        Registers the given value to the class `.register_value` is called from
        :param value: value to register
        :param name: name to register the wrapped value as,
            defaults to value.__name__
        :param alias: alias or list of aliases to register the wrapped value as,
            defaults to None
        """
        register(
            parent_class=cls,
            value=value,
            name=name,
            alias=alias,
            require_subclass=cls.registry_requires_subclass,
        )

    @classmethod
    def load_from_registry(cls, name: str, **constructor_kwargs) -> object:
        """
        :param name: name of registered class to load
        :param constructor_kwargs: arguments to pass to the constructor retrieved
            from the registry
        :return: loaded object registered to this class under the given name,
            constructed with the given kwargs. Raises error if the name is
            not found in the registry
        """
        constructor = cls.get_value_from_registry(name=name)
        return constructor(**constructor_kwargs)

    @classmethod
    def get_value_from_registry(cls, name: str):
        """
        :param name: name to retrieve from the registry
        :return: value from retrieved the registry for the given name, raises
            error if not found
        """
        return get_from_registry(
            parent_class=cls,
            name=name,
            require_subclass=cls.registry_requires_subclass,
        )

    @classmethod
    def registered_names(cls) -> List[str]:
        """
        :return: list of all names registered to this class
        """
        return registered_names(cls)

    @classmethod
    def registered_aliases(cls) -> List[str]:
        """
        :return: list of all aliases registered to this class
        """
        return registered_aliases(cls)


def register(
    parent_class: Type,
    value: Any,
    name: Optional[str] = None,
    alias: Union[List[str], str, None] = None,
    require_subclass: bool = False,
):
    """
    :param parent_class: class to register the name under
    :param value: the value to register
    :param name: name to register the wrapped value as, defaults to value.__name__
    :param alias: alias or list of aliases to register the wrapped value as,
        defaults to None
    :param require_subclass: require that value is a subclass of the class this
        method is called from
    """
    if name is None:
        # default name
        name = value.__name__

    name = standardize_lookup_name(name)
    alias = standardize_alias_name(alias)
    register_alias(name=name, alias=alias, parent_class=parent_class)

    if require_subclass:
        _validate_subclass(parent_class, value)

    if name in _REGISTRY[parent_class]:
        # name already exists - raise error if two different values are attempting
        # to share the same name
        registered_value = _REGISTRY[parent_class][name]
        if registered_value is not value:
            raise RuntimeError(
                f"Attempting to register name {name} as {value} "
                f"however {name} has already been registered as {registered_value}"
            )
    else:
        _REGISTRY[parent_class][name] = value


def get_from_registry(
    parent_class: Type, name: str, require_subclass: bool = False
) -> Any:
    """
    :param parent_class: class that the name is registered under
    :param name: name to retrieve from the registry of the class
    :param require_subclass: require that value is a subclass of the class this
        method is called from
    :return: value from retrieved the registry for the given name, raises
        error if not found
    """
    name = standardize_lookup_name(name)

    if ":" in name:
        # user specifying specific module to load and value to import
        module_path, value_name = name.split(":")
        retrieved_value = _import_and_get_value_from_module(module_path, value_name)
    else:
        # look up name in alias registry
        name = _ALIAS_REGISTRY[parent_class].get(name, name)
        # look up name in registry
        retrieved_value = _REGISTRY[parent_class].get(name)
        if retrieved_value is None:
            raise KeyError(
                f"Unable to find {name} registered under type {parent_class}.\n"
                f"Registered values for {parent_class}: "
                f"{registered_names(parent_class)}\n"
                f"Registered aliases for {parent_class}: "
                f"{registered_aliases(parent_class)}"
            )

    if require_subclass:
        _validate_subclass(parent_class, retrieved_value)

    return retrieved_value


def registered_names(parent_class: Type) -> List[str]:
    """
    :param parent_class: class to look up the registry of
    :return: all names registered to the given class
    """
    return list(_REGISTRY[parent_class].keys())


def registered_aliases(parent_class: Type) -> List[str]:
    """
    :param parent_class: class to look up the registry of
    :return: all aliases registered to the given class
    """
    registered_aliases_plus_names = list(_ALIAS_REGISTRY[parent_class].keys())
    registered_aliases = list(
        set(registered_aliases_plus_names) - set(registered_names(parent_class))
    )
    return registered_aliases


def register_alias(
    name: str, parent_class: Type, alias: Union[str, List[str], None] = None
):
    """
    Updates the mapping from the alias(es) to the given name.
    If the alias is None, the name is used as the alias.
    ```

    :param name: name that the alias refers to
    :param parent_class: class that the name is registered under
    :param alias: single alias or list of aliases that
        refer to the name, defaults to None
    """
    if alias is not None:
        alias = alias if isinstance(alias, list) else [alias]
    else:
        alias = []

    if name in alias:
        raise KeyError(
            f"Attempting to register alias {name}, "
            f"that is identical to the standardized name: {name}."
        )
    alias.append(name)

    for alias_name in alias:
        if alias_name in _ALIAS_REGISTRY[parent_class]:
            raise KeyError(
                f"Attempting to register alias {alias_name} as {name} "
                f"however {alias_name} has already been registered as "
                f"{_ALIAS_REGISTRY[alias_name]}"
            )
        _ALIAS_REGISTRY[parent_class][alias_name] = name


def _import_and_get_value_from_module(module_path: str, value_name: str) -> Any:
    # import the given module path and try to get the value_name if it is included
    # in the module

    # load module
    spec = importlib.util.spec_from_file_location(
        f"plugin_module_for_{value_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # get value from module
    value = getattr(module, value_name, None)

    if not value:
        raise RuntimeError(
            f"Unable to find attribute {value_name} in module {module_path}"
        )
    return value


def _validate_subclass(parent_class: Type, child_class: Type):
    if not issubclass(child_class, parent_class):
        raise ValueError(
            f"class {child_class} is not a subclass of the class it is "
            f"registered for: {parent_class}."
        )
