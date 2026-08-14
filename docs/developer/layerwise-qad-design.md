# LLM Compressor Layerwise QAD Design

## 1. Purpose

This document records the implemented design, key decisions, experiment setup,
results, and known limitations of Layerwise Quantization-Aware Distillation
(Layerwise QAD) in this project. It is intended as an implementation-oriented
reference for maintenance and future experiments.

Core implementation:

- `third_party/llm-compressor/src/llmcompressor/modifiers/layerwise_qad/base.py`
- `third_party/llm-compressor/src/llmcompressor/pipelines/sequential/pipeline.py`
- `third_party/llm-compressor/tests/llmcompressor/modifiers/layerwise_qad/test_base.py`

The generic usage example is in `examples/layerwise_qad/README.md`. The
public-six preparation and benchmark harness used for the reported experiment
were maintained outside this repository and are intentionally not part of the
proposed upstream change.

## 2. Problem and objectives

GPTQ provides a strong quantized initialization without full-model training,
but independent layer reconstruction does not guarantee that the quantized
model remains optimal under accumulated prefix error. Conventional end-to-end
QAD requires complete teacher and student forwards and backpropagation from
final logits. That is expensive and does not fit naturally into LLM
Compressor's sequential calibration pipeline.

This implementation aims to:

1. Reuse GPTQ as the quantized initialization for each block.
2. Train only one decoder block at a time.
3. Use the corresponding full-precision teacher block output as the target.
4. Expose later blocks to accumulated error from all preceding quantized and
   optimized blocks.
5. Freeze quantization parameters and update only quantized Linear master
   weights in the active block.
6. Select the best weight state for every block using held-out validation
   instead of always keeping the final optimizer step.

The method is more precisely described as **quantization-aware block
reconstruction** or **layerwise hidden-state distillation**. It is not
end-to-end QAD based on final-logit KL divergence.

## 3. Core algorithm

For decoder block \(l\), the teacher and student receive the same input
\(X_l\):

\[
Y_l^T = f_l^T(X_l)
\]

\[
Y_l^S = f_l^S(X_l; Q(W_l))
\]

The objective is token-masked hidden-state MSE:

\[
\mathcal{L}_l =
\frac{\sum_{b,t}M_{b,t}
\left(\frac{1}{d}\sum_h(Y^S_{b,t,h}-Y^T_{b,t,h})^2\right)}
{\sum_{b,t}M_{b,t}}
\]

where:

- \(M\) is `loss_mask`, so padding tokens do not contribute;
- \(d\) is the hidden dimension;
- \(Q(W_l)\) is the weight used by the fake-quantized student forward;
- gradients update the floating-point master weight through the fake
  quantizer's straight-through estimator.

After training and restoring the best validation checkpoint, the sequential
pipeline propagates the block's quantized output:

\[
X_{l+1} = f_l^S(X_l; Q(W_l^{best}))
\]

The teacher block intentionally receives the quantized student prefix output,
not the full-precision teacher prefix output. This defines a local conditional
reconstruction problem under the same input and prevents prefix differences
from being mixed into the active block's target.

## 4. Per-block execution

Each decoder block follows this sequence:

1. The sequential pipeline runs calibration forwards for the active block.
2. GPTQ hooks collect Hessian statistics and initialize the block.
3. `LayerwiseQADModifier` captures block arguments and `loss_mask`.
4. All modifier hooks are disabled so QAD forwards cannot alter GPTQ state.
5. The matching teacher block is moved to the student execution device.
6. The student block is checked for initialized scales and fake quantization
   is enabled.
7. All parameters are frozen except quantized Linear weights in the active
   block.
8. Cached batches are deterministically split into training and validation
   sets.
9. Training and validation run per epoch, saving the lowest validation-MSE
   state.
10. Training stops at the epoch limit or by early stopping, then restores the
    best state.
11. Final quantized weights are materialized using the existing scales and
    zero points.
12. The teacher returns to its original device and temporary GPU memory is
    released.
13. The sequential pipeline propagates the optimized quantized output to the
    next block.

## 5. Streaming teacher targets

The first implementation precomputed and cached every teacher hidden-state
target for the active block. This added another dataset-sized hidden-state
cache on top of the input cache.

The current implementation computes the target inside `_batch_loss`:

```python
with torch.no_grad():
    target = teacher_block(*teacher_args, **teacher_kwargs)

prediction = student_block(*args, **kwargs)
loss = masked_mse(prediction, target, loss_mask)
```

The target exists only for the current microbatch and is never stored in
modifier state. Target device memory therefore scales with
`microbatch_size * sequence_length`, not total dataset size.

This is not a fully streaming design. Two input-related representations remain:

- the sequential pipeline's `IntermediatesCache`;
- the active block inputs captured in `LayerwiseQADModifier._layer_inputs`.

This is acceptable for the selected 512x2048 workload. A 22k x 8192 workload
would require chunked replay, shared cache views, or disk-backed storage.

## 6. Validation and model selection

### 6.1 Split

Batch indices are shuffled deterministically with the configured seed:

- 90% training;
- 10% validation;
- validation size is `ceil(batch_count * validation_fraction)`;
- at least one batch remains in both sets.

The formal run uses dataloader batch size 1, producing 461 training batches and
51 validation batches from 512 examples.

### 6.2 Best checkpoint

Before training, the modifier evaluates and snapshots the GPTQ initialization.
After each epoch:

- any strictly lower raw validation loss updates the best weights;
- training always restores the global lowest-loss state;
- a worse final epoch cannot overwrite a better intermediate epoch;
- if no epoch improves, the GPTQ initialization is retained.

### 6.3 Early stopping

Checkpoint selection and patience use separate criteria:

- **Checkpoint criterion:** any lower raw validation loss.
- **Patience reset:** cumulative relative improvement from a reference loss
  reaches `validation_relative_min_delta`.
- **Stop criterion:** no significant cumulative improvement for
  `early_stopping_patience` consecutive epochs.

The formal experiment uses:

```text
num_epochs                    = 12
early_stopping_patience       = 3
validation_relative_min_delta = 0.001
```

This tolerates short-term fluctuations while preventing numerical noise from
extending training indefinitely.

## 7. Training parameter design

The formal public-six experiment uses:

| Parameter | Value | Rationale |
|---|---:|---|
| Optimizer | AdamW | Stable and simple block-local optimizer |
| Learning rate | `1e-5` | Conservative updates around GPTQ initialization |
| Weight decay | `0` | Avoid an unrelated regularization objective |
| Maximum epochs | 12 | Most blocks were still improving after 4 epochs |
| Microbatch size | 1 | Limits memory for 2048-token hidden states |
| Gradient accumulation | 8 | Aggregates eight sequences per update |
| Maximum gradient norm | 1.0 | Bounds abnormal gradients |
| Validation fraction | 0.1 | Provides held-out block selection |
| Seed | 42 | Fixes splits and epoch shuffling |

For 461 training batches, each full epoch performs:

\[
\left\lceil\frac{461}{8}\right\rceil = 58
\]

optimizer steps, or 696 steps for 12 epochs. A final partial accumulation group
is normalized by its actual size and is never discarded.

These values are conservative experiment settings, not universally optimized
defaults. In particular, the modifier API still defaults to learning rate
`2e-6`; the formal runner explicitly selects `1e-5`.

## 8. Dataset design

The final run uses the existing ModelOpt public-six proportions, scaled to 512
examples with sequence length 2048:

| Source | Configuration | Original ratio | Actual examples |
|---|---|---:|---:|
| Nemotron-SWE-v1 | `r2e_gym` | 6000 | 176 |
| Nemotron-Math-v2 | `medium` | 2500 | 73 |
| Nemotron-Science-v1 | `MCQ` | 1500 | 44 |
| Nemotron-Science-v1 | `RQA` | 1500 | 44 |
| Nemotron-Instruction-Following-Chat-v1 | `chat_if` | 5000 | 146 |
| Nemotron-Competitive-Programming-v1 | `competitive_coding_python_part00` | 1000 | 29 |

Artifact statistics:

```text
samples       = 512
stored tokens = 1,048,576
valid tokens  = 764,620
sequence len  = 2048
```

The builder pins every dataset revision and records the configuration SHA256
and input-ID SHA256. `loss_mask` is copied from `attention_mask`, excluding
padding from hidden-state MSE.

## 9. Pipeline integration constraints

When Layerwise QAD is present, `SequentialPipeline` requires:

```text
pipeline                        = sequential
propagate_error                 = true
sequential_targets_per_subgraph = 1
```

`propagate_error=True` ensures that the next block receives the active block's
optimized quantized output. One decoder block per subgraph guarantees an
unambiguous teacher/student mapping and optimization boundary.

AutoRound and Layerwise QAD are currently rejected together because they
require incompatible propagation behavior.

QAD training and propagation forwards disable modifier hooks. This is
essential because GPTQ hooks remain registered for the calibration lifecycle;
without isolation, extra QAD forwards would accumulate Hessian statistics
again and corrupt GPTQ state.

## 10. Formal experiment

Model and quantization:

```text
model             = RedHatAI/Llama-3.1-8B-Instruct
quantization      = NVFP4 W4A4
weight group size = 16
scale dtype       = FP8 E4M3
KV cache          = BF16
ignored module    = lm_head
```

Training outcome:

- one H200 GPU;
- 2:34:43 wall-clock time;
- approximately 7.0 GiB peak GPU memory;
- validation MSE improved for 31 of 32 blocks;
- 28 blocks reached the 12-epoch limit;
- 28 blocks achieved their lowest validation loss at epoch 12;
- those same 28 blocks decreased over their final three epochs;
- `model.layers.1` did not improve and retained its GPTQ initialization.

Fast evaluation with seed 1234:

| Model | MMLU-Pro Chat 5-shot | MATH-500 0-shot |
|---|---:|---:|
| BF16 | 46.76% | 48.40% |
| GPTQ | 44.58% | 43.20% |
| UltraChat 12-epoch QAD | 43.91% | 44.00% |
| public-six 12-epoch QAD | **44.86%** | **46.40%** |

Relative to GPTQ, public-six QAD improves MMLU-Pro by 0.27 percentage points
and MATH-500 by 3.20 points.

The result shows that calibration/QAD data distribution materially affects
downstream quality. It does not prove that learning rate `1e-5` or 12 epochs
is globally optimal.

## 11. Lessons learned

### 11.1 Lower layerwise MSE does not guarantee better tasks

The first UltraChat run reduced MSE for most blocks but scored below GPTQ on
the five-task average. Local hidden-state MSE is not fully aligned with final
logits, generation trajectories, or task accuracy.

### 11.2 Validation-best restoration is required

Always keeping the final step can serialize overfitting or late fluctuations.
Restoring the lowest validation-MSE state is a necessary safeguard, although
it still optimizes only a local proxy.

### 11.3 The epoch limit is a budget, not convergence

In the formal public-six run, 28 blocks were still improving at epoch 12. The
result is the best state in the observed range, not proven convergence.

### 11.4 Data mattered more than simply adding epochs

UltraChat 12-epoch QAD remained below GPTQ on MMLU-Pro. Switching to
public-six moved both MMLU-Pro and MATH-500 above GPTQ. Reports must identify
both optimization settings and dataset composition.

## 12. Limitations and follow-up work

1. **Input cache still scales with dataset size.** Larger datasets need
   chunked or disk-backed replay.
2. **The local objective is imperfect.** Candidates include normalized MSE,
   cosine loss, logits-aware proxies, or short end-to-end refinement.
3. **No systematic LR sweep has been run.** Compare `2e-6`, `5e-6`, and
   `1e-5` while keeping data and checkpoint rules fixed.
4. **Most blocks did not converge.** Raising the epoch limit must be judged by
   both wall-clock cost and downstream metrics.
5. **All blocks share hyperparameters.** Layer-adaptive LR, patience, and
   epoch budgets may be more efficient.
6. **No block-boundary resume exists.** Interrupted runs restart from the
   beginning.
7. **Only matching teacher/student architectures are supported.**
8. **Training is single-GPU.** Multiple GPUs are currently better used for
   parallel hyperparameter experiments.

## 13. Reproduction

See `examples/layerwise_qad/README.md` for the repository-local `oneshot`
example. To reproduce the formal result, use the parameters and pinned dataset
composition recorded above with:

```text
num_epochs                    = 12
learning_rate                 = 1e-5
gradient_accumulation_steps   = 8
validation_fraction           = 0.1
early_stopping_patience       = 3
validation_relative_min_delta = 0.001
seed                          = 42
```

The external experiment harness should additionally record the model revision,
dataset revisions and hashes, quantization configuration, CUDA device, and
serialized output path.
