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
