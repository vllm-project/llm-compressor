from typing import Any, Union

from transformers import PreTrainedTokenizer

# Tokenizer or Processor. Processors do not inherit from a unified base class
Processor = Union[PreTrainedTokenizer, Any]
