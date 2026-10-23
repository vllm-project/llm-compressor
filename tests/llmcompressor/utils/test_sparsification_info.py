import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name", ["helpers", "configs", "module_sparsification_info"]
)
def test_sparsification_info_import(module_name):
    importlib.import_module(
        f"llmcompressor.pytorch.utils.sparsification_info.{module_name}"
    )
