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

Check out our [Developer Guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/) for contributing a new quantization modifier, observer, and more!

## How Can You Contribute?

There are many ways to contribute to LLM Compressor:

- **Reporting Bugs**: If you encounter a bug, please let us know by creating an issue.
- **Suggesting Features**: Have an idea for a new feature? Open an issue to discuss it.
- **Improving Documentation**: Help us improve our documentation by submitting pull requests.
- **Writing Code**: Contribute code to fix bugs, add features, or improve performance.
- **Reviewing Pull Requests**: Provide feedback on open pull requests to help maintain code quality.

## Issue Reporting

If you encounter a bug or have a feature request, please check our issues page first to see if someone else has already reported it.
If not, please file a new issue, providing as much relevant information as possible.

## Claiming Work

We value your time and want to ensure your contributions have the greatest impact.

To make the contribution process as smooth as possible, we ask that you coordinate with maintainers before diving into significant changes. This helps us:

- Avoid duplicate effort if someone else is already working on a similar solution.
- Align on architecture early on so your PR can be merged quickly.
- Protect your time by ensuring the proposed change fits the project's long-term roadmap.

### How to Get Started

1. **Find or create an issue**: Check if your idea is already being discussed. If not, open a new issue to propose the change. Looking at the `good first issue` label is a good way to get involved!
2. **Start a conversation**: Comment on the issue to let us know you're interested! A brief outline of your planned approach is always helpful.
3. **Wait for the "Green Light"**: A maintainer will assign the issue to you. This is our signal that the approach looks good and the "floor is yours."
4. **Build away**: Once assigned, you're all set to begin implementation.

If you haven't heard from us after a week, please feel free to give the thread a nudge!

## RFCs (Request for Comments)

Anyone can contribute to LLM Compressor. For major features, submit an RFC (request for comments) first.
To submit an RFC, create an issue using the feature request template and clearly label it as an RFC.
RFCs are similar to design docs that discuss the motivation, problem solved, alternatives considered, and proposed change.

Once you submit the RFC, please post it in the `#llm-compressor` channel in the [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack), and loop in area owners and committers for feedback.
For high-interest features, the committers nominate a person to help with the RFC process and PR review.
This makes sure someone is guiding you through the process. It is reflected as the "assignee" field in the RFC issue.
If the assignee and lead maintainers find the feature to be contentious, the maintainer team aims to make decisions quickly after learning the details from everyone.
This involves assigning a committer as the DRI (Directly Responsible Individual) to make the decision and shepherd the code contribution process.

## Setup for development

### Install from source

```bash
pip install -e ./[dev]
```

> **Tip:** For development, it is recommended to also install [Compressed Tensors](https://github.com/vllm-project/compressed-tensors) from source:
>
> ```bash
> git clone https://github.com/vllm-project/compressed-tensors.git
> pip install -e ./compressed-tensors
> ```

### Code Styling and Formatting checks

```bash
make style
make quality
```

### Testing

```bash
make test
```

> **Warning:** Running all tests can take a long time and depending on the test might require many GPUs to succeed.

### Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to LLM Compressor.
Your contributions make LLM Compressor a great tool for everyone!
