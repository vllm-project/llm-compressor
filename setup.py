
from setuptools import setup, find_packages
from typing import List, Dict

def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparsification_config", "sparsification_config.*"], exclude=["*.__pycache__.*"]
    )
    
def _setup_install_requires() -> List:
    return ["torch>=1.7.0", "transformers<=4.40", "pydantic>=1.8.2,<2.0.0", "sparsezoo"]

def _setup_extras() -> Dict:
    return {"dev": ["black==22.12.0", "isort==5.8.0", "wheel>=0.36.2", "flake8>=3.8.3"]}

setup(
    name="sparsification_config",
    version="0.0.1",
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description="Library for utilization of compressed safetensors of neural network models",
    extras_require=_setup_extras(),
    install_requires=_setup_install_requires(),
    package_dir={"": "src"},
    packages=_setup_packages(),
)
