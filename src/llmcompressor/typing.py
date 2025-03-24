from typing import Union, Callable

import torch
from datasets import Dataset, DatasetDict, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizer,
    ProcessorMixin,
    PreTrainedModel,
)

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = Union[
    PreTrainedTokenizer, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
]

# Supported dataset types, IterableDataset is a streamed dataset
DatasetType = Union[Dataset, DatasetDict, IterableDataset]

# Pipeline callable
PipelineFn = Callable[[PreTrainedModel, torch.utils.data.DataLoader], None]