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
    name="llmcompressor",
    use_scm_version={
        "version_scheme": version_func,
        "local_scheme": localversion_func,
        "version_file": "src/llmcompressor/version.py",
    },
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description=(
        "A library for compressing large language models utilizing the "
        "latest techniques and research in the field for both "
        "training aware and post training techniques. "
        "The library is designed to be flexible and easy to use on top of "
        "PyTorch and HuggingFace Transformers, allowing for quick experimentation."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "llmcompressor, llms, large language models, transformers, pytorch, "
        "huggingface, compressors, compression, quantization, pruning, "
        "sparsity, optimization, model optimization, model compression, "
    ),
    license="Apache",
    url="https://github.com/vllm-project/llm-compressor",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["llmcompressor", "llmcompressor.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        ("loguru>=0.7.2,<=0.7.3" if BUILD_TYPE == "release" else "loguru>=0.7.2"),
        ("pyyaml>=6.0.1,<=6.0.3" if BUILD_TYPE == "release" else "pyyaml>=6.0.1"),
        # librosa dependency numba is currently not compatible with numpy>=2.3
        # https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information
        ("numpy>=2.0.0,<=2.3.3" if BUILD_TYPE == "release" else "numpy>=2.0.0"),
        (
            "requests>=2.32.2,<=2.32.5"
            if BUILD_TYPE == "release"
            else "requests>=2.32.2"
        ),
        ("tqdm>=4.66.3,<=4.67.1" if BUILD_TYPE == "release" else "tqdm>=4.66.3"),
        ("torch>=2.7.0,<=2.8.0" if BUILD_TYPE == "release" else "torch>=2.7.0"),
        (
            "transformers>=4.53.0,<=4.56.2"
            if BUILD_TYPE == "release"
            else "transformers>=4.53.0"
        ),
        ("datasets>=4.0.0,<=4.1.1" if BUILD_TYPE == "release" else "datasets>=4.0.0"),
        (
            "accelerate>=1.6.0,<=1.10.1"
            if BUILD_TYPE == "release"
            else "accelerate>=1.6.0"
        ),
        (
            "nvidia-ml-py>=12.560.30,<=13.580.82"
            if BUILD_TYPE == "release"
            else "nvidia-ml-py>=12.560.30"
        ),
        ("pillow>=10.4.0,<=11.3.0" if BUILD_TYPE == "release" else "pillow>=10.4.0"),
        (
            "compressed-tensors==0.12.2"
            if BUILD_TYPE == "release"
            else "compressed-tensors>=0.12.3a2"
        ),
    ],
    extras_require={
        "dev": [
            # testing framework
            "pytest>=6.0.0",
            "pytest-mock>=3.6.0",
            "pytest-rerunfailures>=13.0",
            "lm_eval==0.4.5",
            # test dependencies
            "beautifulsoup4~=4.12.3",
            "cmarkgfm~=2024.1.14",
            "trl>=0.10.1",
            "pandas<2.3.0",
            "torchvision",
            "librosa==0.11.0",
            "soundfile",
            "torchcodec",
            # linting, formatting, and type checking
            "mypy~=1.10.0",
            "ruff~=0.4.8",
            # pre commit hooks
            "pre-commit",
            # docs
            "mkdocs",
            "mkdocs-material[imaging]",
            "markdown",
            "pymdown-extensions",
            "mkdocs-section-index",
            "mkdocs-minify-plugin",
            "mkdocs-api-autonav",
            "mkdocstrings-python",
            "mkdocs-gen-files",
            "mkdocs-nav-weight",
        ]
    },
    entry_points={
        "console_scripts": [
            "llmcompressor.trace=llmcompressor.transformers.tracing.debug:main",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
