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
    def __instancecheck__(cls, obj: object) -> bool:
        # Custom PyTorch-aware validation
        if not isinstance(obj, torch.nn.Module):
            return False

        # Example: require registered params/modules by name
        required_parameters = getattr(cls, "__required_parameters__", tuple())
        required_modules = getattr(cls, "__required_modules__", tuple())
        required_buffers = getattr(cls, "__required_buffers__", tuple())
        validate = getattr(cls, "__validate__", lambda obj: True)

        return (
            all(
                name in obj._parameters and obj._parameters[name] is not None
                for name in required_parameters
            )
            and all(
                name in obj._modules and obj._modules[name] is not None
                for name in required_modules
            )
            and all(
                name in obj._buffers and obj._buffers[name] is not None
                for name in required_buffers
            )
            and validate(obj)
        )


class TorchModuleProtocol(Protocol, metaclass=TorchModuleProtocolMeta):
    """
    Base class for torch.nn.Module protocols with custom isinstance behavior.
    """

    __required_parameters__: tuple[str, ...] = ()
    __required_modules__: tuple[str, ...] = ()
    __required_buffers__: tuple[str, ...] = ()
