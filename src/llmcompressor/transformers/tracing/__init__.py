from typing import TYPE_CHECKING

import sys
import importlib
from llmcompressor.utils.AliasableLazyModule import _AliasableLazyModule
from transformers.utils.import_utils import define_import_structure

_aliases = {
    "TraceableLlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),  # noqa: E501
    "TraceableMllamaForConditionalGeneration": ("mllama", "MllamaForConditionalGeneration"),  # noqa: E501
    "TraceableQwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    "TraceableIdefics3ForConditionalGeneration": ("idefics3", "Idefics3ForConditionalGeneration"),  # noqa: E501
    "TraceableWhisperForConditionalGeneration": ("whisper", "WhisperForConditionalGeneration")  # noqa: E501
}

if TYPE_CHECKING:
    for alias, (module_name, class_name) in _aliases.items():
        module = importlib.import_module(f".{module_name}", __package__)
        locals()[alias] = getattr(module, class_name)
else:
    _file = globals()["__file__"]
    sys.modules[__name__] = _AliasableLazyModule(
        name=__name__,
        module_file=_file,
        import_structure=define_import_structure(_file),
        module_spec=__spec__,
        aliases=_aliases
    )

__all__ = list(_aliases.keys())
