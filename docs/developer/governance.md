# Governance

LLM Compressor is part of the [vLLM project](https://github.com/vllm-project) and follows a similar lightweight governance model: a small group of committers is responsible for the technical direction of the project, reviewing and merging contributions, and maintaining the overall health of the codebase.

## Committers and Area Owners

Committers have write access to the LLM Compressor repository and are responsible for:

- Reviewing and merging pull requests.
- Triaging and responding to issues and questions from the community.
- Maintaining the quality, consistency, and direction of the codebase.
- Owning specific areas of the codebase, including reviewing changes in that area, keeping documentation up to date, and guiding related development efforts.

Area ownership is tracked at the file and directory level in the repository's [CODEOWNERS](https://github.com/vllm-project/llm-compressor/blob/main/.github/CODEOWNERS) file, which is the source of truth for which committers are automatically requested for review on a given change.

## Committers

- **[@kylesayrs](https://github.com/kylesayrs)** — Quantization (GPTQ, SmoothQuant, AWQ), pruning (SparseGPT, Wanda, OBCQ), transform-based methods (SpinQuant, QuIP), and the sequential and data-free pipelines.
- **[@dsikka](https://github.com/dsikka)** — Modeling support (MoE and vision model patches), quantization modifiers, observers, transformers compression and data integration, and project release/CI management.
- **[@brian-dellabetta](https://github.com/brian-dellabetta)** — AWQ and transform-based quantization (SpinQuant, QuIP), entrypoints, and lm-eval based accuracy validation.
- **[@HDCharles](https://github.com/HDCharles)** — Quantization and transform modifiers (GPTQ, AWQ, SmoothQuant), observers, and dataset utilities.

New committers are added as they demonstrate sustained, high-quality contributions and take on ownership of a part of the codebase. If you're interested in becoming more involved, start by contributing code, reviews, and issue triage in the areas you care about — see the [Contributing Guide](https://github.com/vllm-project/llm-compressor/blob/main/CONTRIBUTING.md) to get started.
