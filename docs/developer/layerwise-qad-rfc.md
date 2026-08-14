# RFC: Layerwise Hidden-State QAD for LLM Compressor

- **Status:** Experimental / Request for Comments
- **Target project:** `vllm-project/llm-compressor`
- **Reference revision:** `de46bfd53513aa87571a8b056a06aeaa5da1c69c`
- **Proposed component:** `LayerwiseQADModifier`

## Summary

This RFC proposes a decoder-block-wise quantization-aware distillation modifier
for LLM Compressor. After a quantization initializer such as GPTQ completes,
the corresponding full-precision teacher block supervises a fake-quantized
student block through hidden-state MSE. Only quantized floating-point weights
in the active block are updated.

Every block uses held-out validation, early stopping, and best-weight
restoration. Teacher targets are computed on demand per microbatch rather than
materialized for the full dataset. After a block is complete, the sequential
pipeline propagates its optimized quantized output to the next block.

This is not end-to-end QAD based on final-logit KL divergence. More precise
terms are layerwise hidden-state distillation and quantization-aware block
reconstruction.

## 1. Motivation

LLM Compressor already supports:

- sequential decoder-block calibration;
- activation caching and CPU offload;
- quantization modifiers such as GPTQ, AWQ, and AutoRound;
- compressed-tensors fake quantization;
- modifier lifecycles and recipe composition.

What is missing is a block-local distillation primitive that updates a
quantized model's floating-point weights inside the sequential pipeline.
End-to-end QAD provides a more direct objective, but requires complete teacher
and student forwards and a full student backward graph. It is significantly
more expensive and does not naturally match the current calibration lifecycle.

This proposal adds a low-device-memory block reconstruction method that
composes with GPTQ.

## 2. Goals

1. Optimize one decoder block at a time.
2. Give the teacher and student block exactly the same input.
3. Enable fake quantization on every student training forward.
4. Update only weight-quantized floating-point parameters in the active block.
5. Freeze the teacher, scales, zero points, biases, normalization parameters,
   and all other blocks.
6. Support microbatches, gradient accumulation, multiple epochs, and gradient
   clipping.
7. Select the best state for each block using an independent validation split.
8. Propagate the active block's best quantized output to the next block.
9. Compute teacher targets per microbatch without a dataset-sized target cache.
10. Integrate with LLM Compressor recipes, oneshot, and the sequential
    pipeline.

## 3. Non-goals

The first version does not:

- implement final-vocabulary-logit KL divergence;
- jointly optimize multiple decoder blocks;
- revisit a block after advancing to the next block;
- update the teacher;
- learn quantization scales or zero points;
- train embeddings, the LM head, biases, or normalization parameters;
- support architecture-mismatched teachers and students;
- provide a general-purpose fine-tuning Trainer;
- provide data-parallel or model-parallel training;
- fully stream block inputs from the original dataset;
- guarantee that lower local MSE improves downstream accuracy.

## 4. User interface

The modifier should be created in a recipe, while the existing
`distill_teacher` argument provides the teacher:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.layerwise_qad import LayerwiseQADModifier

recipe = [
    GPTQModifier(
        config_groups=quantization_config,
        ignore=["lm_head"],
    ),
    LayerwiseQADModifier(
        num_epochs=12,
        learning_rate=1.0e-5,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        validation_fraction=0.1,
        early_stopping_patience=3,
        validation_relative_min_delta=0.001,
        seed=42,
        target_offload_device="cpu",
    ),
]

oneshot(
    model=student_path,
    distill_teacher=teacher_path,
    recipe=recipe,
    dataset=dataloader,
    pipeline="sequential",
    sequential_targets=["LlamaDecoderLayer"],
    sequential_targets_per_subgraph=1,
    propagate_error=True,
    use_loss_mask=True,
)
```

These values are a validated experiment configuration, not proposed universal
API defaults.

## 5. Configuration

| Field | Type | Current default | Meaning |
|---|---|---:|---|
| `num_epochs` | int | 1 | Maximum epochs for each block |
| `learning_rate` | float | `2e-6` | AdamW learning rate |
| `weight_decay` | float | 0 | AdamW weight decay |
| `gradient_accumulation_steps` | int | 1 | Calibration batches per optimizer update |
| `max_grad_norm` | float/null | 1.0 | Gradient clipping threshold; null disables clipping |
| `seed` | int | 42 | Split and epoch-shuffle seed |
| `target_offload_device` | str | `cpu` | Device for captured inputs and best weights |
| `validation_fraction` | float | 0.1 | Held-out batch fraction |
| `early_stopping_patience` | int | 3 | Epochs without significant cumulative improvement |
| `validation_relative_min_delta` | float | `1e-3` | Relative improvement required to reset patience |

The calibration dataloader's `batch_size` is the QAD microbatch size.

## 6. Mathematical definition

The input to block \(l\) comes from the optimized quantized student prefix:

\[
X_l =
\begin{cases}
X_0, & l=0 \\
f_{l-1}^{S,q}(X_{l-1}), & l>0
\end{cases}
\]

Teacher and student receive the same input:

\[
Y_l^T=f_l^T(X_l)
\]

\[
Y_l^S=f_l^S(X_l;Q(W_l))
\]

The default objective is masked MSE:

\[
\mathcal{L}_l =
\frac{
\sum_{b,t} M_{b,t}
\left(\frac{1}{d}\sum_h(Y^S_{b,t,h}-Y^T_{b,t,h})^2\right)
}{
\sum_{b,t}M_{b,t}
}
\]

Without `loss_mask`, all tokens are valid. An empty mask or a mask that cannot
broadcast to per-token loss must raise an explicit error.

## 7. Algorithm

### 7.1 Initialization

`on_initialize` must:

1. require `State.teacher_model`;
2. reject identical teacher and student instances;
3. infer sequential decoder-block types;
4. map student blocks to teacher blocks by complete module name;
5. verify that corresponding block Python types match;
6. put the teacher in evaluation mode and disable all teacher gradients.

### 7.2 Input capture

`on_calibration_start` registers a `forward_pre` hook on every student decoder
block. The hook stores:

- positional arguments;
- keyword arguments;
- the current calibration batch's `loss_mask`.

All tensors are detached and moved to `target_offload_device`.

### 7.3 Ordering with GPTQ

The intended recipe order is:

```text
GPTQModifier -> LayerwiseQADModifier
```

At `sequential_epoch_end`, GPTQ must finish the active block's weight and
quantization-parameter initialization before Layerwise QAD begins. If a
quantized weight scale is unavailable, QAD must fail instead of silently
falling back to unquantized fine-tuning.

### 7.4 Block-local optimization

For the active block:

1. find modules that have weight quantization, an initialized scale, and a
   floating-point `weight`;
2. freeze all student block parameters;
3. enable gradients only for those weights;
4. move the teacher block to the student execution device;
5. enable fake quantization on the student block;
6. evaluate initial training and validation MSE;
7. train with AdamW one epoch at a time;
8. validate and update the best checkpoint after each epoch;
9. stop early or reach `num_epochs`;
10. restore the best checkpoint;
11. materialize quantized weights using existing quantization parameters;
12. freeze the student block and restore the teacher's original device.

### 7.5 Streaming teacher targets

The target is computed inside every `_batch_loss`:

```python
with torch.no_grad():
    target = extract_hidden(teacher_block(*teacher_args, **teacher_kwargs))

prediction = extract_hidden(student_block(*args, **kwargs))
loss = masked_mse(prediction, target, loss_mask)
```

The implementation must not retain the complete teacher-target dataset in
modifier state. Floating-point teacher inputs are converted to the teacher
dtype, while integer masks, position IDs, and similar metadata retain their
original dtype.

### 7.6 Gradient accumulation

For \(N\) training batches and accumulation factor \(G\):

\[
\text{steps per epoch}=\left\lceil\frac{N}{G}\right\rceil
\]

The final group must execute even when it contains fewer than \(G\) batches.
Its loss is normalized by the actual group size:

```python
(loss / len(accumulation_group)).backward()
```

### 7.7 Validation and early stopping

The split is deterministic from `seed`, and all blocks reuse the same
batch-index rule.

The GPTQ state and its validation loss are the initial checkpoint. After each
epoch:

```text
if validation_loss < best_validation_loss:
    save best weights

relative_improvement =
    (patience_reference_loss - validation_loss)
    / abs(patience_reference_loss)

if relative_improvement >= relative_min_delta:
    patience_reference_loss = validation_loss
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
```

Raw-loss checkpoint selection and significant-improvement patience must remain
separate. The lowest raw validation-loss weights are restored on exit.

### 7.8 Propagation

After optimization, the pipeline reruns the quantized block over the complete
cache in stable sample order and updates the next subgraph's inputs. It must
not use the final training microbatch output as the next block's cache.

## 8. Sequential pipeline contract

When `LayerwiseQADModifier` is present, the pipeline must enforce:

```text
propagate_error                 = true
sequential_targets_per_subgraph = 1
```

The current version rejects simultaneous AutoRound and Layerwise QAD because
they require incompatible error-propagation behavior.

QAD-generated forwards and propagation forwards must execute inside
`HooksMixin.disable_hooks()`. This prevents GPTQ and other modifier hooks from
collecting statistics again.

## 9. State and observability

The modifier exposes defensive copies of:

- `optimizer_steps: dict[str, int]`
- `best_validation_losses: dict[str, float]`
- `epochs_completed: dict[str, int]`
- `validation_histories: dict[str, list[float]]`

Every epoch logs:

- block name;
- current validation MSE;
- best validation MSE;
- current patience count.

Every completed block logs:

- optimizer steps;
- completed epochs;
- initial and final training MSE;
- initial and final validation MSE.

## 10. Memory model

Current memory consists of:

1. the student model and active student block;
2. the active teacher block;
3. the sequential `IntermediatesCache`;
4. captured active-block inputs;
5. the current microbatch teacher target;
6. optimizer state for the active block;
7. the active block's best-weight snapshot.

Teacher-target device memory does not scale with total calibration examples.
Input caching still grows as:

\[
O(N \times L \times d)
\]

where \(N\) is sequence count, \(L\) is sequence length, and \(d\) is hidden
dimension. The first version therefore supports workloads whose input cache
fits available host memory.

## 11. Error handling

The implementation must fail explicitly when:

- no teacher is supplied;
- teacher and student are the same instance;
- no sequential decoder blocks are found;
- the teacher lacks a corresponding block;
- corresponding block types differ;
- a sequential subgraph contains multiple decoder blocks;
- fewer than two calibration batches are available;
- the active block has no trainable quantized floating-point weights;
- quantization scales are uninitialized;
- teacher block parameters span multiple devices;
- teacher and student hidden-state shapes differ;
- the mask cannot broadcast or contains no valid tokens;
- `propagate_error=False`;
- AutoRound and Layerwise QAD are enabled together.

There must be no success-shaped fallback to unquantized training, unmasked
loss, or validation-free model selection.

## 12. Compatibility

The first version assumes:

- a decoder-only model;
- matching teacher/student block names and types;
- block outputs represented as a Tensor, tuple/list first element, or
  `last_hidden_state`/`hidden_states` in a mapping or object;
- a differentiable compressed-tensors fake-quantized forward;
- weight scales and zero points initialized by a preceding modifier.

Llama 3.1 8B with NVFP4 W4A4 has been validated on a real model. Other
architectures and quantization formats require independent qualification.

## 13. Testing requirements

### 13.1 Unit tests

1. Masked MSE excludes padding.
2. Empty masks and shape mismatches fail explicitly.
3. The final partial accumulation group is updated.
4. Validation splitting is deterministic.
5. Only quantized weights in the active block update.
6. Scales, zero points, and inactive blocks remain unchanged.
7. Best-checkpoint restoration rolls back a degraded epoch.
8. Patience tolerates fluctuations and uses cumulative relative improvement.
9. Teacher targets are generated inside `_batch_loss`.
10. Captured block inputs are released after completion.

### 13.2 Integration tests

1. GPTQ and Layerwise QAD execute in order on the same block.
2. QAD forwards do not accumulate additional GPTQ Hessian statistics.
3. Real fake-quantized training reduces hidden-state MSE.
4. The pipeline rejects invalid propagation settings.
5. `oneshot(distill_teacher=...)` populates the compression session.
6. The modifier factory discovers `LayerwiseQADModifier`.
7. Export and reload preserve materialized quantized weights.

## 14. Validated result

On Llama 3.1 8B Instruct, NVFP4 W4A4, and public-six 512x2048:

- all 32 decoder blocks completed;
- validation MSE improved for 31 of 32 blocks;
- one H200 completed the run in 2:34:43;
- peak GPU memory was approximately 7.0 GiB;
- MMLU-Pro Chat improved from GPTQ 44.58% to 44.86%;
- MATH-500 improved from GPTQ 43.20% to 46.40%.

This demonstrates functional correctness and a quality improvement for one
configuration. It does not establish universal gains across models, datasets,
or tasks.

## 15. Risks

### 15.1 Local-objective mismatch

Lower block MSE does not guarantee better final logits or generation accuracy.
Release decisions require downstream benchmarks.

### 15.2 Quantization-bin stagnation

Small master-weight updates may not cross quantization-bin boundaries, leaving
materialized weights unchanged. Learning rate and step count must be validated
for each quantization format.

### 15.3 Calibration overfitting

Many epochs on a small cache may improve held-out block MSE without improving
task generalization. Multi-seed downstream evaluation is required.

### 15.4 Incomplete convergence

Most blocks in the formal run were still improving at the epoch limit. A
higher limit costs more and may not improve downstream metrics.

### 15.5 Host memory

Streaming teacher targets remove the target cache but not the complete input
cache. Larger datasets require stricter streaming replay.

## 16. Alternatives

### 16.1 End-to-end logits KL

This aligns more directly with final behavior but requires a full student
backward graph and complete teacher forward.

### 16.2 Independent teacher and student activation streams

This follows each model's natural trajectory but mixes prefix differences into
the active block objective and requires two activation caches.

### 16.3 Joint multi-layer hidden-state losses

This may reduce local-objective mismatch but loses the sequential block-local
memory advantage.

### 16.4 Rounding-parameter-only optimization

AutoRound already covers this direction. This proposal updates floating-point
master weights under a fixed quantization scheme.

## 17. Rollout

1. Keep the API experimental and avoid cross-architecture guarantees.
2. Retain toy-model, GPTQ-composition, and pipeline-contract tests in CI.
3. Add export/reload coverage on a small real transformer.
4. Qualify fake-quantization gradients and materialization per format.
5. Implement chunked input replay before scaling calibration data.
6. Stabilize naming and API only after multi-model, multi-seed evaluation.

## 18. Open questions

1. Should the public name retain QAD or use block reconstruction?
2. Should normalized MSE, cosine loss, or mixed losses be supported?
3. Should learning rate, epoch budget, and patience adapt by layer?
4. Should bias or normalization parameters ever be trainable?
5. Is block-boundary checkpoint and resume required?
6. How should duplicate storage between `IntermediatesCache` and
   `_layer_inputs` be removed?
7. Can short end-to-end refinement correct local-objective mismatch?
8. Which quantization formats can formally support differentiable weight
   updates?
