# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup, find_packages
from typing import List, Dict, Tuple


# Set the build type using an environment variable to give us
# different package names based on the reason for the build.
VALID_BUILD_TYPES = {"release", "nightly", "dev"}
BUILD_TYPE = os.environ.get("BUILD_TYPE", "dev")
if BUILD_TYPE not in VALID_BUILD_TYPES:
    raise ValueError(
        f"Unsupported build type {BUILD_TYPE!r}, must be one of {VALID_BUILD_TYPES}"
    )

from setuptools_scm import ScmVersion

def version_func(version: ScmVersion) -> str:
    from setuptools_scm.version import guess_next_version

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


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"

def _setup_packages() -> List:
    return find_packages(
        "src", include=["compressed_tensors", "compressed_tensors.*"], exclude=["*.__pycache__.*"]
    )

def _setup_install_requires() -> List:
    return ["torch>=1.7.0", "transformers", "pydantic>=2.0", "frozendict"]

def _setup_extras() -> Dict:
    return {
        "dev": ["black==22.12.0", "isort==5.8.0", "wheel>=0.36.2", "flake8>=3.8.3", "pytest>=6.0.0", "nbconvert>=7.16.3"],
        "accelerate": ["accelerate"]
    }

setup(
    name="compressed-tensors",
    use_scm_version={
        "version_scheme": version_func,
        "local_scheme": localversion_func,
        "version_file": "src/compressed_tensors/version.py",
    },
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    license="Apache 2.0",
    description="Library for utilization of compressed safetensors of neural network models",
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    url="https://github.com/neuralmagic/compressed-tensors",
    extras_require=_setup_extras(),
    install_requires=_setup_install_requires(),
    package_dir={"": "src"},
    package_data={"": ["transform/utils/hadamards.safetensors"]},
    packages=_setup_packages(),
)
