import os
import sys

from setuptools import find_packages, setup

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils.version_extractor import extract_version_info  # noqa isort:skip

# load version info for the package
package_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "src", "llmcompressor"
)
version_info = extract_version_info(package_path)

if version_info.build_type == "release":
    package_name = "llmcompressor"
elif version_info.build_type == "dev":
    package_name = "llmcompressor-dev"
elif version_info.build_type == "nightly":
    package_name = "llmcompressor-nightly"
else:
    raise ValueError(f"Unsupported build type {version_info.build_type}")


setup(
    name=package_name,
    version=version_info.version,
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
    url="https://github.com/neuralmagic/llm-compressor",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["llmcompressor", "llmcompressor.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        "loguru",
        "pyyaml>=5.0.0",
        "numpy>=1.17.0,<2.0",
        "requests>=2.0.0",
        "tqdm>=4.0.0",
        "click>=7.1.2,!=8.0.0",  # 8.0.0 blocked due to reported bug
        "torch>=1.7.0",
        "transformers>4.0,<5.0",
        "datasets",
        "accelerate>=0.20.3",
        "pynvml==11.5.3",
        "compressed-tensors"
        if version_info.build_type == "release"
        else "compressed-tensors-nightly",
    ],
    extras_require={
        "dev": [
            # testing
            "pytest>=6.0.0",
            "pytest-mock>=3.6.0",
            "pytest-rerunfailures>=13.0",
            "parameterized",
            # linting, formatting, and type checking
            "black~=24.4.2",
            "isort~=5.13.2",
            "mypy~=1.10.0",
            "ruff~=0.4.8",
            "flake8~=7.0.0",
            # pre commit hooks
            "pre-commit",
        ]
    },
    entry_points={
        "console_scripts": [
            "llmcompressor.transformers.text_generation.apply=llmcompressor.transformers.finetune.text_generation:apply",  # noqa 501
            "llmcompressor.transformers.text_generation.compress=llmcompressor.transformers.finetune.text_generation:apply",  # noqa 501
            "llmcompressor.transformers.text_generation.train=llmcompressor.transformers.finetune.text_generation:train",  # noqa 501
            "llmcompressor.transformers.text_generation.finetune=llmcompressor.transformers.finetune.text_generation:train",  # noqa 501
            "llmcompressor.transformers.text_generation.eval=llmcompressor.transformers.finetune.text_generation:eval",  # noqa 501
            "llmcompressor.transformers.text_generation.oneshot=llmcompressor.transformers.finetune.text_generation:oneshot",  # noqa 501
        ]
    },
    python_requires=">=3.8",
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
