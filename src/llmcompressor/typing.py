from typing import TYPE_CHECKING, Callable, Union

import torch
from datasets import Dataset, DatasetDict, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = Union[
    PreTrainedTokenizer, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
]

# Supported dataset types, IterableDataset is a streamed dataset
DatasetType = Union[Dataset, DatasetDict, IterableDataset]

# Pipeline callable
PipelineFn = Callable[
    [PreTrainedModel, torch.utils.data.DataLoader, "DatasetArguments"], None
]
