from typing import TYPE_CHECKING

import sys
from transformers.utils import _LazyModule

_import_structure = {
    "llava": ["LlavaForConditionalGeneration"],
    "mllama": ["MllamaForConditionalGeneration"],
    "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    "idefics3": ["Idefics3ForConditionalGeneration"],
    "whisper": ["WhisperForConditionalGeneration"],
    "qwen2_audio": ["Qwen2AudioForConditionalGeneration"],
}

if TYPE_CHECKING:
    from .llava import LlavaForConditionalGeneration as TraceableLlavaForConditionalGeneration  # noqa: E501
    from .mllama import MllamaForConditionalGeneration as TraceableMllamaForConditionalGeneration  # noqa: E501
    from .qwen2_vl import Qwen2VLForConditionalGeneration as TraceableQwen2VLForConditionalGeneration  # noqa: E501
    from .idefics3 import Idefics3ForConditionalGeneration as TraceableIdefics3ForConditionalGeneration  # noqa: E501
else:
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        _import_structure,
        module_spec=__spec__,
        extra_objects={
            "TraceableLlavaForConditionalGeneration": "llava.LlavaForConditionalGeneration",  # noqa: E501
            "TraceableMllamaForConditionalGeneration": "mllama.MllamaForConditionalGeneration",  # noqa: E501
            "TraceableQwen2VLForConditionalGeneration": "qwen2_vl.Qwen2VLForConditionalGeneration",  # noqa: E501
            "TraceableIdefics3ForConditionalGeneration": "idefics3.Idefics3ForConditionalGeneration",  # noqa: E501
        },
    )

__all__ = [
    "TraceableLlavaForConditionalGeneration",
    "TraceableMllamaForConditionalGeneration",
    "TraceableQwen2VLForConditionalGeneration",
    "TraceableIdefics3ForConditionalGeneration",
]
