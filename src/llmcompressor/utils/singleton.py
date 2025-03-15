from typing import Optional, Type, TypeVar
from abc import ABC

T = TypeVar("T", bound="SingletonMixin")

class SingletonMixin(ABC):
    _instances: dict[Type["SingletonMixin"], "SingletonMixin"] = {}

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        else:
            raise ValueError(f"Cannot construct more than one instance of {cls}")

        return cls._instances[cls]

    @classmethod
    def instance(cls: Type[T]) -> T:
        if cls not in cls._instances:
            raise ValueError(f"Instance of {cls} has not been created yet.")
        return cls._instances[cls]