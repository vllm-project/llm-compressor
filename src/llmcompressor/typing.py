from typing import TYPE_CHECKING, Callable, List, Union

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
    from llmcompressor.args.post_train_arguments import PostTrainArguments
    from llmcompressor.modifiers import Modifier

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = Union[
    PreTrainedTokenizer, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
]

# Supported dataset types, IterableDataset is a streamed dataset
DatasetType = Union[Dataset, DatasetDict, IterableDataset]

# Pipeline callable
PipelineFn = Callable[
    [PreTrainedModel, torch.utils.data.DataLoader, "PostTrainArguments"], None
]

# Supported model input types
ModelInput = Union[str, PreTrainedModel]

# Supported recipe input types
RecipeInput = Union[str, "Modifier", List["Modifier"]]
