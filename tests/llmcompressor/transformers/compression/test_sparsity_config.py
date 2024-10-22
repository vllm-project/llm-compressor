import pytest

from llmcompressor.transformers.compression.sparsity_config import SparsityStructure


def test_sparsity_structure_valid_cases():
    assert (
        SparsityStructure("2:4") == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4' with TWO_FOUR"
    assert (
        SparsityStructure("unstructured") == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'unstructured' with UNSTRUCTURED"
    assert (
        SparsityStructure("UNSTRUCTURED") == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'UNSTRUCTURED' with UNSTRUCTURED"
    assert (
        SparsityStructure(None) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match None with UNSTRUCTURED"


def test_sparsity_structure_invalid_case():
    with pytest.raises(ValueError, match="invalid is not a valid SparsityStructure"):
        SparsityStructure("invalid")


def test_sparsity_structure_case_insensitivity():
    assert (
        SparsityStructure("2:4") == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4' with TWO_FOUR"
    assert (
        SparsityStructure("2:4".upper()) == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4'.upper() with TWO_FOUR"
    assert (
        SparsityStructure("unstructured".upper()) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'unstructured'.upper() with UNSTRUCTURED"
    assert (
        SparsityStructure("UNSTRUCTURED".lower()) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'UNSTRUCTURED'.lower() with UNSTRUCTURED"


def test_sparsity_structure_default_case():
    assert (
        SparsityStructure(None) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match None with UNSTRUCTURED"
