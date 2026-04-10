import os
import sys

from setuptools import find_packages, setup
from setuptools_scm import ScmVersion

# Set the build type using an environment variable to give us
# different package names based on the reason for the build.
VALID_BUILD_TYPES = {"release", "nightly", "dev"}
BUILD_TYPE = os.environ.get("BUILD_TYPE", "dev")
if BUILD_TYPE not in VALID_BUILD_TYPES:
    raise ValueError(
        f"Unsupported build type {BUILD_TYPE!r}, must be one of {VALID_BUILD_TYPES}"
    )


def version_func(version: ScmVersion) -> str:
    from setuptools_scm.version import guess_next_version

    print(
        f"computing version for {BUILD_TYPE} build with "
        f"{'dirty' if version.dirty else 'clean'} local repository"
        f"{' and exact version from tag' if version.exact else ''}",
        file=sys.stderr,
    )

    if BUILD_TYPE == "nightly":
        # Nightly builds use alpha versions to ensure they are marked
        # as pre-releases on pypi.org.
        return version.format_next_version(
            guess_next=guess_next_version,
            fmt="{guessed}.a{node_date:%Y%m%d}",
        )

    if (
        BUILD_TYPE == "release"
        and not version.dirty
        and (version.exact or version.node is None)
    ):
        # When we have a tagged version, use that without modification.
        return version.format_with("{tag}")

    # In development mode or when the local repository is dirty, treat
    # it is as local development version.
    return version.format_next_version(
        guess_next=guess_next_version,
        fmt="{guessed}.dev{distance}",
    )


def localversion_func(version: ScmVersion) -> str:
    from setuptools_scm.version import get_local_node_and_date

    print(
        f"computing local version for {BUILD_TYPE} build with "
        f"{'dirty' if version.dirty else 'clean'} local repository"
        "f{' and exact version from tag' if version.exact else ''}",
        file=sys.stderr,
    )

    # When we are building nightly versions, we guess the next release
    # and add the date as an alpha version. We cannot publish packages
    # with local versions, so we do not add one.
    if BUILD_TYPE == "nightly":
        return ""

    # When we have an exact tag, with no local changes, do not append
    # anything to the local version field.
    if (
        BUILD_TYPE == "release"
        and not version.dirty
        and (version.exact or version.node is None)
    ):
        return ""

    # In development mode or when the local repository is dirty,
    # return a string that includes the git SHA (node) and a date,
    # formatted as a local version tag.
    return get_local_node_and_date(version)


setup(
    use_scm_version={
        "version_scheme": version_func,
        "local_scheme": localversion_func,
        "version_file": "src/llmcompressor/version.py",
    },
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["llmcompressor", "llmcompressor.*"], exclude=["*.__pycache__.*"]
    ),
)
