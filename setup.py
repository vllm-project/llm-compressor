"""
Thin setup.py retained for setuptools_scm version computation and
BUILD_TYPE-dependent dependency pinning for release / nightly builds.

All static project metadata has moved to pyproject.toml (PEP 621).
For local development you can simply run:

    pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"

Release and nightly CI pipelines should set BUILD_TYPE=release or
BUILD_TYPE=nightly before invoking the build.
"""

import os
import sys

from setuptools import setup
from setuptools_scm import ScmVersion

# ---- build-type gating -----------------------------------------------------
VALID_BUILD_TYPES = {"release", "nightly", "dev"}
BUILD_TYPE = os.environ.get("BUILD_TYPE", "dev")
if BUILD_TYPE not in VALID_BUILD_TYPES:
    raise ValueError(
        f"Unsupported build type {BUILD_TYPE!r}, must be one of {VALID_BUILD_TYPES}"
    )


# ---- version helpers --------------------------------------------------------
def version_func(version: ScmVersion) -> str:
    from setuptools_scm.version import guess_next_version

    print(
        f"computing version for {BUILD_TYPE} build with "
        f"{'dirty' if version.dirty else 'clean'} local repository"
        f"{' and exact version from tag' if version.exact else ''}",
        file=sys.stderr,
    )

    if BUILD_TYPE == "nightly":
        return version.format_next_version(
            guess_next=guess_next_version,
            fmt="{guessed}.a{node_date:%Y%m%d}",
        )

    if (
        BUILD_TYPE == "release"
        and not version.dirty
        and (version.exact or version.node is None)
    ):
        return version.format_with("{tag}")

    return version.format_next_version(
        guess_next=guess_next_version,
        fmt="{guessed}.dev{distance}",
    )


def localversion_func(version: ScmVersion) -> str:
    from setuptools_scm.version import get_local_node_and_date

    if BUILD_TYPE == "nightly":
        return ""
    if (
        BUILD_TYPE == "release"
        and not version.dirty
        and (version.exact or version.node is None)
    ):
        return ""
    return get_local_node_and_date(version)


# ---- release dependency pins ------------------------------------------------
# For release builds we pin upper bounds.  For dev / nightly the
# lower-bound-only specifiers from pyproject.toml are used as-is.

_RELEASE_OVERRIDES: dict[str, str] = {
    "loguru": "loguru>=0.7.2,<=0.7.3",
    "pyyaml": "pyyaml>=6.0.1,<=6.0.3",
    "numpy": "numpy>=2.0.0,<=2.4.2",
    "requests": "requests>=2.32.2,<=2.32.5",
    "tqdm": "tqdm>=4.66.3,<=4.67.3",
    "torch": "torch>=2.9.0,<=2.10.0",
    "datasets": "datasets>=4.0.0,<=4.6.0",
    "auto-round": "auto-round>=0.9.6,<=0.10.2",
    "accelerate": "accelerate>=1.6.0,<=1.12.0",
    "nvidia-ml-py": "nvidia-ml-py>=12.560.30,<=13.590.48",
    "pillow": "pillow>=10.4.0,<=12.1.1",
    "compressed-tensors": "compressed-tensors==0.14.0",
}


def _build_install_requires() -> list[str] | None:
    """Return overridden deps for release builds; None otherwise."""
    if BUILD_TYPE != "release":
        return None  # fall back to pyproject.toml dependencies
    return list(_RELEASE_OVERRIDES.values())


# ---- setup() ----------------------------------------------------------------
setup(
    use_scm_version={
        "version_scheme": version_func,
        "local_scheme": localversion_func,
        "version_file": "src/llmcompressor/version.py",
    },
    install_requires=_build_install_requires(),
)
