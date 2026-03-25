from unittest.mock import patch

import torch
from compressed_tensors.offload.cache import OffloadCache

from llmcompressor.modeling.moe_context import (
    _apply_offloading_to_replacement,
    _find_ancestor_with_offload_cache,
)


def test_find_ancestor_with_offload_cache():
    """Test finding ancestor modules with OffloadCache."""
    # Module without offload cache
    module_no_cache = torch.nn.Linear(10, 10)
    assert _find_ancestor_with_offload_cache(module_no_cache) is None

    # Module with offload cache
    module_with_cache = torch.nn.Linear(10, 10)
    module_with_cache._parameters = OffloadCache()
    assert _find_ancestor_with_offload_cache(module_with_cache) is module_with_cache

    # Parent with child that has cache
    parent = torch.nn.Sequential(module_with_cache)
    assert _find_ancestor_with_offload_cache(parent) is module_with_cache


@patch("llmcompressor.modeling.moe_context.get_cache_init_kwargs")
@patch("llmcompressor.modeling.moe_context.offload_module")
def test_apply_offloading_to_replacement(mock_offload, mock_get_kwargs):
    """Test offloading is applied from original to replacement."""
    mock_get_kwargs.return_value = {"device": "cpu"}

    # Original with offload cache
    original = torch.nn.Sequential(torch.nn.Linear(10, 10))
    original[0]._parameters = OffloadCache()

    # Replacement without cache
    replacement = torch.nn.Sequential(torch.nn.Linear(10, 10))

    _apply_offloading_to_replacement(original, replacement)

    # Should call offload_module for the child linear layer
    assert mock_offload.called
    assert mock_get_kwargs.called


def test_apply_offloading_no_cache():
    """Test no offloading applied when original has no cache."""
    original = torch.nn.Linear(10, 10)
    replacement = torch.nn.Linear(10, 10)

    # Should not raise, just return early
    _apply_offloading_to_replacement(original, replacement)
