"""
Module for generating and managing version information for LLM Compressor.

This module provides functionality for creating semantic version strings based on the
version base and build type. It supports `release`, `nightly`, and `dev` build types.
"""

from typing import Optional, Tuple

# Define the base version and build type
version_base = "0.3.0"
build_type = "dev"  # can be 'release', 'nightly', 'dev', or 'dev' with a dev number


def _generate_version_attributes(
    base, type_
) -> Tuple[str, int, int, int, Optional[str]]:
    from datetime import datetime

    parts = base.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if type_ == "release":
        build = None
    elif type_ == "nightly":
        build = datetime.utcnow().strftime("%Y%m%d")
    elif "dev" in type_:
        build = type_
    else:
        raise ValueError(f"Unknown build type: {type_}")

    ver = f"{major}.{minor}.{patch}"
    if build:
        ver += f".{build}"

    return ver, major, minor, patch, build


version, version_major, version_minor, version_patch, version_build = (
    _generate_version_attributes(version_base, build_type)
)
__version__ = version


__all__ = [
    "__version__",
    "version_base",
    "build_type",
    "version",
    "version_major",
    "version_minor",
    "version_patch",
    "version_build",
]
