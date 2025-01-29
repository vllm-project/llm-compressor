import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_imports():
    # Remove any existing imports before each test
    module_names = list(sys.modules.keys())
    for module_name in module_names:
        if module_name.startswith("llmcompressor"):
            del sys.modules[module_name]

    importlib.invalidate_caches()

    yield


@pytest.mark.unit
def test_lazy_loading(clean_imports):
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

    # test that llava was imported and mllama was not
    assert ".llava" in imported_module_names
    assert ".mllama" not in imported_module_names

    # test that tracing module has a llava attribute but not an mllama attribute
    attributes = ModuleType.__dir__(sys.modules["llmcompressor.transformers.tracing"])
    assert "llava" in attributes
    assert "mllama" not in attributes


@pytest.mark.unit
def test_class_names(clean_imports):
    import llmcompressor.transformers.tracing as TracingModule

    # test that the class names are not the aliased names
    # this is important for correctly saving model configs
    for cls_alias, (_loc, cls_name) in TracingModule._aliases.items():
        cls = getattr(TracingModule, cls_alias)
        assert cls.__name__ == cls_name
