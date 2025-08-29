"""
Defines type aliases for the llm-compressor library, specifically Processor (tokenizer and processor types) and DatasetType (supported dataset formats).
"""

from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizer,
    ProcessorMixin,
)

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = Union[
    PreTrainedTokenizer, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
]

# Supported dataset types, IterableDataset is a streamed dataset
DatasetType = Union[Dataset, DatasetDict, IterableDataset]
