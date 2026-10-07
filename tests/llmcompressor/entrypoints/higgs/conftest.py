"""Fixtures shared by HIGGS entrypoint tests."""

import pytest


@pytest.fixture
def require_higgs_cbc():
    """Skip solver-dependent tests when PuLP or its CBC executable is absent."""
    try:
        import pulp
    except ImportError:
        pytest.skip("PuLP is required for HIGGS ILP tests")

    bundled_cbc_path = getattr(
        getattr(pulp, "PULP_CBC_CMD", None), "pulp_cbc_path", None
    )
    try:
        available = pulp.COIN_CMD(path=bundled_cbc_path).available()
    except (AttributeError, OSError):
        available = None

    if not available:
        pytest.skip("CBC is required for HIGGS ILP tests")
