"""
Defines type aliases for the llm-compressor library.
"""

from typing import Iterable

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
