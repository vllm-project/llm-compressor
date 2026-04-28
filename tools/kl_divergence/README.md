# KL Divergence via Hidden State Extraction

Efficiently measure KL divergence between a base model and a quantized/compressed
model using vLLM's hidden state extraction API.

## Background

KL divergence measures how a model's output distribution differs from a reference.
Extracting full-vocab logprobs from vLLM is prohibitively slow (~1.1 tok/s for
Llama-3-8B, ~64 hours for wikitext) because serializing 128k-dim vectors per token
is the bottleneck.

**This tool extracts hidden states before the lm_head instead** (dim ~4096 vs
vocab ~128k = ~30x smaller), then computes logprobs and KL divergence offline.

## Requirements

- **Phase 1** (extraction): `vllm >= 0.18.0`, `transformers`, `datasets`, `safetensors`
- **Phase 2** (computation): `torch`, `safetensors`, `transformers` (no vLLM needed)

## Usage

### Phase 1: Extract Hidden States

Run for both the base model and the quantized model:

```bash
# Base model
python tools/kl_divergence/extract_hidden_states.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir ./hidden_states/base \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --max-seq-length 2048 \
    --num-samples 128

# Quantized model
python tools/kl_divergence/extract_hidden_states.py \
    --model ./Meta-Llama-3-8B-Instruct-W4A16 \
    --output-dir ./hidden_states/quantized \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --max-seq-length 2048 \
    --num-samples 128
```

### Phase 2: Compute KL Divergence

```bash
python tools/kl_divergence/compute_kl.py \
    --base-dir ./hidden_states/base \
    --target-dir ./hidden_states/quantized \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --target-model ./Meta-Llama-3-8B-Instruct-W4A16 \
    --temperature 1.0 \
    --device cuda:0 \
    --output results.json
```

If the base and quantized models share the same lm_head (typical for weight-only
quantization), `--target-model` can be omitted and the base model's lm_head is used
for both.

## Options

### extract_hidden_states.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | HuggingFace model ID or local path |
| `--output-dir` | (required) | Where to save hidden states |
| `--dataset` | `Salesforce/wikitext` | HuggingFace dataset name |
| `--dataset-config` | `wikitext-2-raw-v1` | Dataset configuration |
| `--split` | `test` | Dataset split |
| `--max-seq-length` | `2048` | Token chunk size |
| `--num-samples` | all | Max number of chunks to process |
| `--layer-index` | auto | Layer to extract (default: last) |
| `--gpu-memory-utilization` | `0.9` | vLLM GPU memory fraction |
| `--tensor-parallel-size` | `1` | vLLM tensor parallelism |
| `--text-column` | `text` | Dataset text column name |

### compute_kl.py

| Option | Default | Description |
|--------|---------|-------------|
| `--base-dir` | (required) | Base model hidden states directory |
| `--target-dir` | (required) | Target model hidden states directory |
| `--base-model` | (required) | Base model ID (for lm_head weights) |
| `--target-model` | same as base | Target model ID (if lm_head differs) |
| `--temperature` | `1.0` | Softmax temperature |
| `--device` | `cuda:0` | Computation device |
| `--chunk-size` | `64` | Tokens per chunk (lower = less memory) |
| `--output` | stdout | Path for results JSON |
| `--norm-weight-name` | auto | Override final norm tensor name |
| `--lm-head-weight-name` | auto | Override lm_head tensor name |
| `--lm-head-bias-name` | auto | Override lm_head bias tensor name |
| `--embed-weight-name` | auto | Override embed tensor name |

## Supported Architectures

Architecture-specific tensor names are auto-detected. The following are supported
out of the box:

- **Llama family**: Llama, Llama-2, Llama-3, CodeLlama
- **Qwen family**: Qwen2, Qwen3
- **Mistral**
- **Gemma family**: Gemma, Gemma-2, Gemma-3
- **Phi family**: Phi, Phi-3
- **Others**: Granite, InternLM2, Cohere, DeepSeek V3

For unsupported architectures, use the `--lm-head-weight-name` and
`--embed-weight-name` overrides. The default (`lm_head.weight`) works for most
modern LLMs.

## Limitations

- **vLLM version**: Phase 1 requires `vllm >= 0.18.0` with the hidden state extraction
  API. This API is relatively new and may change across vLLM versions.
- **Post-norm extraction**: Hidden states are extracted after the final layer norm
  (at layer index `num_hidden_layers`), so norm is not re-applied during computation.
  This means the tool works regardless of norm type (RMSNorm, LayerNorm, etc.).
- **Same tokenization required**: Base and target models must be extracted using the
  same tokenizer and dataset configuration. Token IDs are validated during computation.
- **Prompt-only**: vLLM hidden state extraction captures prompt token hidden states
  only, not generated tokens. This is correct for KL divergence evaluation.

## How It Works

1. **Extract**: vLLM's hidden state extraction API (built on Eagle-3 speculative
   decoding infrastructure) captures the output of a specified transformer layer
   and saves it as safetensors files. Only the last layer is needed for KL divergence.

2. **Compute**: The saved hidden states (post-norm) are loaded and the lm_head
   projection is applied to reconstruct full-vocab logits. KL divergence is then
   computed per-position and aggregated.

The key advantage is that hidden states (dim ~4096) are ~30x smaller than logprobs
(vocab ~128k), making extraction and storage dramatically faster and cheaper.

## Output Format

The results JSON contains:

```json
{
  "mean_kl": 0.012345,
  "mean_kl_per_sample": 0.012400,
  "std_kl": 0.003456,
  "median_kl": 0.011234,
  "min_kl": 0.001234,
  "max_kl": 0.034567,
  "num_samples": 128,
  "total_tokens": 262144,
  "temperature": 1.0,
  "elapsed_seconds": 45.2,
  "tokens_per_second": 5800,
  "per_sample_kl": [0.012, 0.013, ...],
  "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "target_model": "./Meta-Llama-3-8B-Instruct-W4A16",
  "dataset": "Salesforce/wikitext"
}
```
