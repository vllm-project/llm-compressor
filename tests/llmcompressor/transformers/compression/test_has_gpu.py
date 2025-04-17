import os

import pytest
import torch


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true", reason="Only run for GHA")
def test_has_gpu():
    """
    This test exists purely to raise an error if
    a runner performs transformers tests without a GPU
    """
    assert torch.cuda.is_available()
