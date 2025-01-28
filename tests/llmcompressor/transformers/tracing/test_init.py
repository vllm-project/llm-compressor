import sys
from types import ModuleType

import importlib
from unittest.mock import patch
from transformers.utils import _LazyModule

def test_lazy_loading():
    # from llmcompressor.transformers.tracing import TraceableLlavaForConditionalGeneration
    
    # attributes = ModuleType.__dir__(sys.modules["llmcompressor.transformers.tracing"])
    # breakpoint()
    # assert "llava" in attributes
    # assert "mllama" not in attributes


    imported_module_names = []
    original_import_module = importlib.import_module
    def mock_import_module(name, *args, **kwargs):
        nonlocal imported_module_names
        imported_module_names.append(name)
        return original_import_module(name, *args, **kwargs)

    with patch('transformers.utils._LazyModule.importlib.import_module', side_effect=mock_import_module):
        from llmcompressor.transformers.tracing import TraceableLlavaForConditionalGeneration

    breakpoint()
    assert "llmcompressor.transformers.tracing.llava" in imported_module_names