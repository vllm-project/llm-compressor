
from setuptools import setup, find_packages
from typing import List

def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparsification_config", "sparsification_config.*"], exclude=["*.__pycache__.*"]
    )
    
setup(
    name="sparsification_config",
    version="0.0.1",
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description="Library for manipulating sparse safetensors of neural network models",
    install_requires=["torch>=1.7.0", "transformers<=4.40", "pydantic>=1.8.2,<2.0.0", "sparsezoo", "black==22.12.0", "isort==5.8.0", "wheel>=0.36.2", "flake8>=3.8.3"],
    #packages=_setup_packages(),
)
