import importlib
import pkgutil
from typing import Dict, Type

from llmcompressor.modifiers.modifier import Modifier

__all__ = ["ModifierFactory"]


class ModifierFactory:
    """
    A factory for loading and registering modifiers
    """

    _MAIN_PACKAGE_PATH = "llmcompressor.modifiers"
    _EXPERIMENTAL_PACKAGE_PATH = "llmcompressor.modifiers.experimental"

    _loaded: bool = False
    _main_registry: Dict[str, Type[Modifier]] = {}
    _experimental_registry: Dict[str, Type[Modifier]] = {}
    _registered_registry: Dict[str, Type[Modifier]] = {}
    _errors: Dict[str, Exception] = {}

    @staticmethod
    def refresh():
        """
        A method to refresh the factory by reloading the modifiers
        Note: this will clear any previously registered modifiers
        """
        ModifierFactory._main_registry = ModifierFactory.load_from_package(
            ModifierFactory._MAIN_PACKAGE_PATH
        )
        ModifierFactory._experimental_registry = ModifierFactory.load_from_package(
            ModifierFactory._EXPERIMENTAL_PACKAGE_PATH
        )
        ModifierFactory._loaded = True

    @staticmethod
    def load_from_package(package_path: str) -> Dict[str, Type[Modifier]]:
        """
        :param package_path: The path to the package to load modifiers from
        :return: The loaded modifiers, as a mapping of name to class
        """
        loaded = {}
        main_package = importlib.import_module(package_path)

        # exclude deprecated packages from registry so
        # their new location is used instead
        deprecated_packages = [
            "llmcompressor.modifiers.obcq",
            "llmcompressor.modifiers.obcq.sgpt_base",
        ]
        for _importer, modname, _is_pkg in pkgutil.walk_packages(
            main_package.__path__, package_path + "."
        ):
            if modname in deprecated_packages:
                continue
            try:
                module = importlib.import_module(modname)

                for attribute_name in dir(module):
                    if not attribute_name.endswith("Modifier"):
                        continue

                    try:
                        if attribute_name in loaded:
                            continue

                        attr = getattr(module, attribute_name)

                        if not isinstance(attr, type):
                            raise ValueError(
                                f"Attribute {attribute_name} is not a type"
                            )

                        if not issubclass(attr, Modifier):
                            raise ValueError(
                                f"Attribute {attribute_name} is not a Modifier"
                            )

                        loaded[attribute_name] = attr
                    except Exception as err:
                        # TODO: log import error
                        ModifierFactory._errors[attribute_name] = err
            except Exception as module_err:
                # TODO: log import error
                print(module_err)

        return loaded

    @staticmethod
    def create(
        type_: str,
        allow_registered: bool,
        allow_experimental: bool,
        **kwargs,
    ) -> Modifier:
        """
        Instantiate a modifier of the given type from registered modifiers.

        :raises ValueError: If no modifier of the given type is found
        :param type_: The type of modifier to create
        :param framework: The framework the modifier is for
        :param allow_registered: Whether or not to allow registered modifiers
        :param allow_experimental: Whether or not to allow experimental modifiers
        :param kwargs: Additional keyword arguments to pass to the modifier
            during instantiation
        :return: The instantiated modifier
        """
        if type_ in ModifierFactory._errors:
            raise ModifierFactory._errors[type_]

        if type_ in ModifierFactory._registered_registry:
            if allow_registered:
                return ModifierFactory._registered_registry[type_](**kwargs)
            else:
                # TODO: log warning that modifier was skipped
                pass

        if type_ in ModifierFactory._experimental_registry:
            if allow_experimental:
                return ModifierFactory._experimental_registry[type_](**kwargs)
            else:
                # TODO: log warning that modifier was skipped
                pass

        if type_ in ModifierFactory._main_registry:
            return ModifierFactory._main_registry[type_](**kwargs)

        raise ValueError(f"No modifier of type '{type_}' found.")

    @staticmethod
    def register(type_: str, modifier_class: Type[Modifier]):
        """
        Register a modifier class to be used by the factory.

        :raises ValueError: If the provided class does not subclass the Modifier
            base class or is not a type
        :param type_: The type of modifier to register
        :param modifier_class: The class of the modifier to register, must subclass
            the Modifier base class
        """
        if not issubclass(modifier_class, Modifier):
            raise ValueError(
                "The provided class does not subclass the Modifier base class."
            )
        if not isinstance(modifier_class, type):
            raise ValueError("The provided class is not a type.")

        ModifierFactory._registered_registry[type_] = modifier_class
