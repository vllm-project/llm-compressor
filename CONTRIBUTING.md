# Contributing to LLM Compressor

Thank you for your interest in contributing to LLM Compressor!
Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large.
There are several ways you can contribute to the project:

- Identify and report any issues or bugs.
- Request or add new compression methods or research.
- Suggest or implement new features.

However, remember that contributions aren't just about code.
We believe in the power of community support; thus, answering queries, assisting others, and enhancing the documentation are highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about LLM Compressor and the vLLM community.
Talk about it in your blog posts, highlighting how it's driving your incredible projects.
Express your support on Twitter if vLLM aids you, or simply offer your appreciation by starring our repository.

## Setup for development

### Install from source

```bash
pip install -e ./[dev]
```

!!! tip
    For development, it is recommended to also install [Compressed Tensors](https://github.com/vllm-project/compressed-tensors) from source:

    ```bash
    git clone https://github.com/vllm-project/compressed-tensors.git
    pip install -e ./compressed-tensors
    ```

### Code Styling and Formatting checks

```bash
make style
make quality
```

### Testing

```bash
make test
```

!!! warning
    Running all tests can take a long time and depending on the test might require many GPUs to succeed.

## Contributing Guidelines

### Issue Reporting

If you encounter a bug or have a feature request, please check our issues page first to see if someone else has already reported it.
If not, please file a new issue, providing as much relevant information as possible.

### Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to LLM Compressor.
Your contributions make LLM Compressor a great tool for everyone!
