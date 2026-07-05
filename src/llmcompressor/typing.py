"""
Defines type aliases for the llm-compressor library.
"""

from typing import Iterable, Protocol, _ProtocolMeta

import torch
from datasets import Dataset, DatasetDict, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizer,
    ProcessorMixin,
)

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = (
    PreTrainedTokenizer | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin
)

# Supported dataset types, IterableDataset is a streamed dataset
DatasetType = Dataset | DatasetDict | IterableDataset

# Torch types
NamedModules = Iterable[tuple[str, torch.nn.Module]]


class TorchModuleProtocolMeta(_ProtocolMeta):
    """
    Metaclass that implements custom isinstance checks for torch.nn.Module protocols.

    Subclasses must implement a __validate__ classmethod that checks if an object
    satisfies the protocol's requirements.
    """

    def __instancecheck__(cls: "TorchModuleProtocol", obj: object) -> bool:
        return isinstance(obj, torch.nn.Module) and cls.__validate__(obj)


class TorchModuleProtocol(Protocol, metaclass=TorchModuleProtocolMeta):
    """
    Base protocol class for torch.nn.Module subclasses with custom isinstance behavior.

    Subclasses should implement a __validate__ classmethod to define custom validation
    logic that will be checked during isinstance() calls.

    Example:
        ```
        class MyModuleProtocol(TorchModuleProtocol):
            weight: torch.nn.Parameter

            @classmethod
            def __validate__(cls, obj: object) -> bool:
                return isinstance(getattr(obj, "weight", None), torch.nn.Parameter)
        ```

    Note: While protocols cannot be recognized as torch.nn.Module subclasses at the
    type level (they use structural typing), instances validated with isinstance()
    are guaranteed to be torch.nn.Module instances.
    """

    @classmethod
    def __validate__(cls, obj: object) -> bool:
        """
        Validate that an object satisfies this protocol's requirements.

        Args:
            obj: The object to validate

        Returns:
            True if the object satisfies the protocol, False otherwise
        """
        return True
