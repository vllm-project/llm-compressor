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
from typing import Dict, List, Tuple

from setuptools import find_packages, setup
from utils.artifacts import get_release_and_version


package_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "src", "sparseml"
)
(
    is_release,
    is_dev,
    version,
    version_major,
    version_minor,
    version_bug,
) = get_release_and_version(package_path)

# load and overwrite version and release info from sparseml package
exec(open(os.path.join("src", "sparseml", "version.py")).read())
print(f"loaded version {version} from src/sparseml/version.py")
version_nm_deps = f"{version_major}.{version_minor}.0"

if is_release:
    _PACKAGE_NAME = "sparseml"
elif is_dev:
    _PACKAGE_NAME = "sparseml-dev"
else:
    _PACKAGE_NAME = "sparseml-nightly"

_deps = [
    "pyyaml>=5.0.0",
    "numpy>=1.17.0,<2.0",
    "matplotlib>=3.0.0",
    "merge-args>=0.1.0",
    "onnx>=1.5.0,<1.15.0",
    "pandas>=0.25.0",
    "packaging>=20.0",
    "psutil>=5.0.0",
    "pydantic>=2.0.0,<2.8.0",
    "requests>=2.0.0",
    "scikit-learn>=0.24.2",
    "scipy<1.9.2,>=1.8; python_version <= '3.9'",
    "scipy>=1.0.0; python_version > '3.9'",
    "tqdm>=4.0.0",
    "toposort>=1.0",
    "GPUtil>=1.4.0",
    "protobuf>=3.12.2,<=3.20.3",
    "click>=7.1.2,!=8.0.0",  # latest version < 8.0 + blocked version with reported bug
]
_nm_deps = [f"{'sparsezoo' if is_release else 'sparsezoo-nightly'}>=1.7.0"]
_deepsparse_deps = [f"{'deepsparse' if is_release else 'deepsparse-nightly'}>=1.7.0"]
_deepsparse_ent_deps = ["deepsparse-ent>=1.7.0"]

_onnxruntime_deps = ["onnxruntime>=1.0.0"]
_clip_deps = ["open_clip_torch==2.20.0"]
supported_torch_version = "torch>=1.7.0"
_pytorch_deps = [
    supported_torch_version,
    "gputils",
]
_pytorch_all_deps = _pytorch_deps + [
    "torchvision>=0.3.0,<0.17",
    "torchaudio<=2.0.1",
]
_pytorch_vision_deps = _pytorch_deps + [
    "torchvision>=0.3.0,<0.17",
    "opencv-python<=4.6.0.66",
]
_transformers_deps = _pytorch_deps + [
    "transformers<4.41",
    "datasets<2.19",
    "dvc",
    "scikit-learn",
    "seqeval",
    "einops",
    "evaluate>=0.4.1",
    "accelerate>=0.20.3",
    "safetensors>=0.4.1",
    "compressed-tensors" if is_release else "compressed-tensors-nightly",
]
_llm_deps = _transformers_deps + ["sentencepiece"]
_yolov5_deps = _pytorch_vision_deps + [
    f"{'nm-yolov5' if is_release else 'nm-yolov5-nightly'}<={version_nm_deps}"
]
_notebook_deps = [
    "jupyter>=1.0.0",
    "ipywidgets>=7.0.0",
]
_tensorflow_v1_deps = ["tensorflow<2.0.0", "tensorboard<2.0.0", "tf2onnx>=1.0.0,<1.6"]
_tensorflow_v1_gpu_deps = [
    "tensorflow-gpu<2.0.0",
    "tensorboard<2.0.0",
    "tf2onnx>=1.0.0,<1.6",
]
_keras_deps = ["tensorflow~=2.2.0", "keras2onnx>=1.0.0"]

_open_pif_paf_deps = ["openpifpaf==0.13.6"]

_dev_deps = [
    "beautifulsoup4==4.9.3",
    "black==22.12.0",
    "flake8==3.9.2",
    "isort==5.8.0",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "pytest-mock>=3.6.0",
    "pytest-rerunfailures>=13.0",
    "tensorboard>=1.0,<2.9",
    "tensorboardX>=1.0",
    "evaluate>=0.4.1",
    "parameterized",
]

_docs_deps = [
    "m2r2>=0.2.7",
    "mistune<3,>=2.0.3",
    "myst-parser>=0.14.0",
    "rinohtype~=0.4.2",
    "sphinx~=3.5.0",
    "sphinx-copybutton~=0.3.0",
    "sphinx-markdown-tables~=0.0.15",
    "sphinx-multiversion~=0.2.4",
    "sphinx-pydantic~=0.1.0",
    "sphinx-rtd-theme~=0.5.0",
    "docutils<0.17",
]


_ultralytics_deps = [
    "ultralytics==8.0.124",
    supported_torch_version,
]


def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparseml", "sparseml.*"], exclude=["*.__pycache__.*"]
    )


def _setup_package_dir() -> Dict:
    return {"": "src"}


def _setup_install_requires() -> List:
    return _nm_deps + _deps


def _setup_extras() -> Dict:
    return {
        "clip": _clip_deps,
        "dev": _dev_deps,
        "docs": _docs_deps,
        "deepsparse": _deepsparse_deps,
        "deepsparse-ent": _deepsparse_ent_deps,
        "openpifpaf": _open_pif_paf_deps,
        "onnxruntime": _onnxruntime_deps,
        "torch": _pytorch_deps,
        "torch_all": _pytorch_all_deps,
        "torchvision": _pytorch_vision_deps,
        "transformers": _transformers_deps,
        "llm": _llm_deps,
        "notebook": _notebook_deps,
        "tf_v1": _tensorflow_v1_deps,
        "tf_v1_gpu": _tensorflow_v1_gpu_deps,
        "tf_keras": _keras_deps,
        "ultralytics": _ultralytics_deps,
        "yolov5": _yolov5_deps,
    }


def _setup_entry_points() -> Dict:
    entry_points = {
        "console_scripts": [
            "sparseml.transformers.text_generation.apply=sparseml.transformers.finetune.text_generation:apply",  # noqa 501
            "sparseml.transformers.text_generation.compress=sparseml.transformers.finetune.text_generation:apply",  # noqa 501
            "sparseml.transformers.text_generation.train=sparseml.transformers.finetune.text_generation:train",  # noqa 501
            "sparseml.transformers.text_generation.finetune=sparseml.transformers.finetune.text_generation:train",  # noqa 501
            "sparseml.transformers.text_generation.eval=sparseml.transformers.finetune.text_generation:eval",  # noqa 501
            "sparseml.transformers.text_generation.oneshot=sparseml.transformers.finetune.text_generation:oneshot",  # noqa 501
        ]
    }

    return entry_points


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name=_PACKAGE_NAME,
    version=version,
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description=(
        "Libraries for applying sparsification recipes to neural networks with a "
        "few lines of code, enabling faster and smaller models"
    ),
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords=(
        "inference, machine learning, neural network, computer vision, nlp, cv, "
        "deep learning, torch, pytorch, tensorflow, keras, sparsity, pruning, "
        "deep learning libraries, onnx, quantization, automl"
    ),
    license="Apache",
    url="https://github.com/neuralmagic/sparseml",
    include_package_data=True,
    package_dir=_setup_package_dir(),
    packages=_setup_packages(),
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.8.0,<3.12",
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
