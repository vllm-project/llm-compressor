from datetime import datetime

import pytest

from llmcompressor import (
    build_type,
    version,
    version_base,
    version_build,
    version_major,
    version_minor,
    version_patch,
)
from llmcompressor.version import _generate_version_attributes


@pytest.mark.smoke
def test_version_attributes():
    ver, major, minor, patch, build = _generate_version_attributes(
        version_base, build_type
    )
    assert ver == version
    assert major == version_major
    assert minor == version_minor
    assert patch == version_patch
    assert build == version_build


@pytest.mark.sanity
def test_release_version():
    version_base = "1.8.0"
    build_type = "release"
    ver, major, minor, patch, build = _generate_version_attributes(
        version_base, build_type
    )
    assert ver == "1.8.0"
    assert build is None


@pytest.mark.sanity
def test_nightly_version():
    version_base = "1.8.0"
    build_type = "nightly"
    ver, major, minor, patch, build = _generate_version_attributes(
        version_base, build_type
    )
    assert ver == f"1.8.0.{datetime.utcnow().strftime('%Y%m%d')}"
    assert build == datetime.utcnow().strftime("%Y%m%d")


@pytest.mark.sanity
def test_dev_version():
    version_base = "1.8.0"
    build_type = "dev1"
    ver, major, minor, patch, build = _generate_version_attributes(
        version_base, build_type
    )
    assert ver == "1.8.0.dev1"
    assert build == "dev1"


@pytest.mark.sanity
def test_invalid_version():
    version_base = "1.8.0"
    build_type = "unknown"
    with pytest.raises(ValueError):
        _generate_version_attributes(version_base, build_type)
