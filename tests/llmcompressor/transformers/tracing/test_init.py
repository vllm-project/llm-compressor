import importlib
import sys
from types import ModuleType
from unittest.mock import patch


def test_lazy_loading():
    # mock import_module
    imported_module_names = []
    original_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        nonlocal imported_module_names
        imported_module_names.append(name)
        return original_import_module(name, *args, **kwargs)

    # import with alias
    with patch("importlib.import_module", mock_import_module):
        from llmcompressor.transformers.tracing import (  # noqa: F401
            TraceableLlavaForConditionalGeneration,
        )

    # test that llava was imported by mllama was not
    assert ".llava" in imported_module_names
    assert ".mllama" not in imported_module_names

    # test that tracing module has a llava attribute but not an mllama attribute
    attributes = ModuleType.__dir__(sys.modules["llmcompressor.transformers.tracing"])
    assert "llava" in attributes
    assert "mllama" not in attributes


def test_class_names():
    import llmcompressor.transformers.tracing as TracingModule

    # test that the class names are not the aliased names
    # this is important for correctly saving model configs
    for cls_alias, (_loc, cls_name) in TracingModule._aliases.items():
        cls = getattr(TracingModule, cls_alias)
        assert cls.__name__ == cls_name
