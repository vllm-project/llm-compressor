---
weight: -10
---

# Installation

LLM Compressor can be installed using several methods depending on your requirements. Below are the detailed instructions for each installation pathway.

## Prerequisites

Before installing LLM Compressor, ensure you have the following prerequisites:

- **Operating System:** Linux (recommended for GPU support)
- **Python Version:** 3.9 or newer
- **Pip Version:** Ensure you have the latest version of pip installed. You can upgrade pip using the following command:

  ```bash
  python -m pip install --upgrade pip
  ```

## Installation Methods

### Install from PyPI

The simplest way to install LLM Compressor is via pip from the Python Package Index (PyPI):

```bash
pip install llmcompressor
```

This will install the latest stable release of LLM Compressor.

### Install a Specific Version from PyPI

If you need a specific version of LLM Compressor, you can specify the version number during installation:

```bash
pip install llmcompressor==0.5.1
```

Replace `0.5.1` with your desired version number.

### Install from Source

To install the latest development version of LLM Compressor from the main branch, use the following command:

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git
```

This will clone the repository and install LLM Compressor directly from the main branch.

### Install from a Local Clone

If you have cloned the LLM Compressor repository locally and want to install it, navigate to the repository directory and run:

```bash
pip install .
```

For development purposes, you can install it in editable mode with the `dev` extra:

```bash
pip install -e .[dev]
```

This allows you to make changes to the source code and have them reflected immediately without reinstalling.
