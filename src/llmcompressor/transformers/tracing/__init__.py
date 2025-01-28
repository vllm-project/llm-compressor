from typing import TYPE_CHECKING

import sys
from llmcompressor.utils.import_utils import _AliasableLazyModule
from transformers.utils.import_utils import define_import_structure

__all__ = [
    "TraceableLlavaForConditionalGeneration",
    "TraceableMllamaForConditionalGeneration",
    "TraceableQwen2VLForConditionalGeneration",
    "TraceableIdefics3ForConditionalGeneration",
]

if TYPE_CHECKING:
    from .llava import LlavaForConditionalGeneration as TraceableLlavaForConditionalGeneration  # noqa: E501
    from .mllama import MllamaForConditionalGeneration as TraceableMllamaForConditionalGeneration  # noqa: E501
    from .qwen2_vl import Qwen2VLForConditionalGeneration as TraceableQwen2VLForConditionalGeneration  # noqa: E501
    from .idefics3 import Idefics3ForConditionalGeneration as TraceableIdefics3ForConditionalGeneration  # noqa: E501
else:
    _file = globals()["__file__"]
    sys.modules[__name__] = _AliasableLazyModule(
        name=__name__,
        module_file=_file,
        import_structure=define_import_structure(_file),
        module_spec=__spec__,
        aliases={
            "TraceableLlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),  # noqa: E501
            "TraceableMllamaForConditionalGeneration": ("mllama", "MllamaForConditionalGeneration"),  # noqa: E501
            "TraceableQwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
            "TraceableIdefics3ForConditionalGeneration": ("idefics3", "Idefics3ForConditionalGeneration"),  # noqa: E501
        }
    )
