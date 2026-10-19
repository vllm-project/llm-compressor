from types import SimpleNamespace
from unittest.mock import patch

from tests.testing_utils import requires_compute_capability


def test_requires_compute_capability_skips_non_cuda_accelerator():
    """CUDA SM requirements must skip cleanly on XPU-style backends."""
    with (
        patch("torch.accelerator.is_available", return_value=True),
        patch(
            "torch.accelerator.current_accelerator",
            return_value=SimpleNamespace(type="xpu"),
        ),
    ):
        marker = requires_compute_capability(9, 0)

    assert marker.mark.name == "skip"
    assert marker.mark.kwargs["reason"] == (
        "CUDA compute capability required, found xpu"
    )
