# Subgraph quantization-aware distillation

`QADModifier` jointly optimizes quantized floating-point weights in each traced
sequential subgraph. Put a quantization modifier before it in the recipe:
`QuantizationModifier` for RTN, or `GPTQModifier` for GPTQ.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Choose either initialization method.
quantizer = QuantizationModifier(scheme="NVFP4A16", ignore=["lm_head"])  # RTN
# quantizer = GPTQModifier(scheme="NVFP4A16", ignore=["lm_head"], actorder="static")
qad = QADModifier(
    teacher_mode="local",  # "full" uses original-model inputs at every boundary
    num_epochs=12,
    learning_rate=2e-6,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    validation_fraction=0.1,
    early_stopping_patience=3,
    validation_relative_min_delta=0.001,
)

# model is an original model; dataset contains tokenized samples.
oneshot(
    model=model,
    dataset=dataset,
    recipe=[quantizer, qad],
    pipeline="sequential",
    sequential_targets=["LlamaDecoderLayer"],
    sequential_targets_per_subgraph=2,
    propagate_error=True,
    batch_size=1,
    num_calibration_samples=512,
    max_seq_length=2048,
)
```

See [llama3_example.py](llama3_example.py) for a complete NVFP4A16 example:

```bash
python llama3_example.py --quantizer rtn --output ./llama-rtn-qad
python llama3_example.py --quantizer gptq --output ./llama-gptq-qad
```

## Separate PTQ and QAD datasets

`dataset` supplies the preceding quantizer's calibration data. Optionally pass
`qad_dataset` for QAD training and its internal validation split:

```python
oneshot(
    model=model,
    processor=tokenizer,
    recipe=[quantizer, qad],
    pipeline="sequential",
    dataset=gptq_dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
    batch_size=1,
    qad_dataset=qad_dataset,
    qad_dataset_args={
        "num_calibration_samples": 1024,
        "max_seq_length": 1024,
        "batch_size": 2,
        "shuffle_calibration_samples": False,
    },
)
```

Both inputs accept Hugging Face dataset names, `Dataset`, `DatasetDict`, or a
pre-built PyTorch `DataLoader`. `qad_dataset_args` accepts `DatasetArguments`
keywords such as `splits`, `dataset_config_name`, `preprocessing_func`, and
`text_column`. Its defaults are independent of the PTQ arguments: set each
dataset's sample limit, preprocessing and batch settings explicitly as needed.
A supplied `DataLoader` controls its own batching and preprocessing.
Pipeline/tracing settings remain those of the main `oneshot` call.

Each stream propagates through the final quantized/QAD weights independently.
Only `dataset` triggers PTQ calibration hooks. QAD captures its teacher before
current-subgraph quantization. Student inputs from `qad_dataset` include preceding
subgraphs' quantization error; teacher inputs follow `teacher_mode`. No sample pairing
between the two datasets is required. Both must work with the same traced model
graph and input structure; batch size and sequence length may differ.

With a separate dataset, QAD uses its own `loss_mask` when provided, independently
of the primary dataset's `use_loss_mask` setting. It needs at least two QAD
batches for its training/validation split. An additional intermediate activation
cache and propagation forwards are required for the separate stream.

For disjoint UltraChat subsets (first 512 rows for GPTQ, next 1024 for QAD):

```bash
python llama3_example.py --quantizer gptq --samples 512 \
  --qad-dataset HuggingFaceH4/ultrachat_200k --qad-offset 512 \
  --qad-samples 1024 --output ./llama-gptq-qad-separate
```

For another dataset, set `--qad-dataset`, `--qad-split`, and `--qad-text-column`
as appropriate. The example uses `messages` with the model's chat template when
available, otherwise the specified text column.

Omitting `qad_dataset` retains the original shared-data behavior without a second
cache. Separation does not deduplicate datasets: choose disjoint samples yourself
if QAD's validation data must also be unseen by GPTQ. The internal validation
split selects QAD weights/early stopping; downstream test data should stay separate.

## Execution and teacher

Before calibrating the current subgraph, QAD runs its unquantized computation with
modifier hooks disabled and caches teacher outputs. The preceding quantizer then
calibrates that subgraph. QAD trains against the cached outputs, and the pipeline
propagates the optimized quantized outputs to the next subgraph. Teacher and
student inputs depend on `QADModifier(teacher_mode=...)`:

| Mode | Teacher target for subgraph i | Student input |
| --- | --- | --- |
| `"local"` (default) | Original current weights applied to the preceding final quantized outputs | Preceding final quantized outputs |
| `"full"` | Original current weights applied to the preceding original-model outputs | Preceding final quantized outputs |

With original subgraph function `f_i`, original boundary values `h_i`, and student
boundary values `h_hat_i`, the targets are `f_i(h_hat_{i-1})` for local mode and
`f_i(h_{i-1})` for full mode. Both start from the same tokenized sample. QAD still
trains one traced subgraph at a time, with its own early stopping and fixed qparams.

Full mode maintains an independent teacher activation cache from the input,
including subgraphs without quantized weights. Before each stage's PTQ/QAD update,
one original-weight forward supplies both current loss targets and all boundary
values needed by later stages. It reuses weights that have not yet been modified;
no second model is loaded and no past stage is rerun with changed weights.
The extra cache uses `target_offload_device` (default CPU). Consumed values are
released as the teacher advances, and the cache is released after the last stage.
This adds activation storage and transfer; stages without QAD also need teacher
propagation. The exact cost depends on the traced graph and dataset.

Both modes work with shared or separate PTQ/QAD datasets. Full mode's original
stream follows only QAD samples when `qad_dataset` is provided. Switching modes
does not change the PTQ calibration stream or enable qparam re-observation.

```python
qad = QADModifier(teacher_mode="full", num_epochs=3, learning_rate=2e-6)
# Or retain the original behavior:
qad = QADModifier(teacher_mode="local", num_epochs=3, learning_rate=2e-6)
```

The complete script accepts `--teacher-mode full` or `--teacher-mode local`.

Branches, residual additions, repeated module calls and multiple sequential
targets retain their traced computation. Floating output leaves affected by
quantized weights contribute equally to the mean of output MSEs. Other outputs
and metadata do not contribute. Calibration disables the output LM head; QAD
reconstructs the ignored head's input hidden states instead of vocabulary logits.
If the terminal pipeline subgraph exports no values, QAD observes its terminal
weight-dependent values explicitly.
Both modes reconstruct subgraph hidden outputs; they do not use vocabulary-logit
distillation or backpropagate through preceding subgraphs.

## Training

- `batch_size` (in `qad_dataset_args` when using separate data) is the microbatch
  size. Each optimizer step combines
  `gradient_accumulation_steps` microbatches; a final partial group is averaged
  by its actual size.
- Each subgraph has a deterministic training/validation split and its own AdamW
  optimizer. `seed` controls the split and epoch shuffling.
- Any lower validation MSE saves the best weights. Patience resets only after
  cumulative relative improvement reaches `validation_relative_min_delta`.
  Initial weights are eligible for restoration.
- Qparams remain fixed during QAD and final weight materialization. There is no
  re-observation or learned scale optimization. `NVFP4A16` leaves activations in
  the model's floating dtype; the complete example explicitly loads FP16.
- Inputs, teacher outputs and best-weight snapshots use `target_offload_device`
  (default CPU), and are released after each subgraph. FP16/BF16 execution weights
  use FP32 optimizer master weights/state to retain small updates.
- With shared data, `use_loss_mask=True` in `oneshot` uses dataset-provided masks;
  with separate data, QAD uses its own `loss_mask` when present. Masks must
  broadcast to each floating output's shape without its last feature dimension,
  e.g. `[batch, tokens]` for `[batch, tokens, hidden]` outputs.
- Results are exposed through `optimizer_steps`, `epochs_completed`,
  `best_validation_losses`, and `validation_histories`.

## Current boundaries

Use the sequential pipeline with `propagate_error=True`, at least two calibration
batches, and an original model with the quantizer before QAD in the recipe.
Keep the output LM head ignored. Uses of a shared quantized weight must reside in
the same subgraph: an earlier update would invalidate the later original teacher.
RTN and GPTQ are covered by tests. Other quantizers must preserve original weights
until the subgraph-start event and initialize qparams before QAD's end callback.
Full mode additionally requires that earlier modifiers never change weights or
representations used by future teacher stages. Combining full mode with smoothing
or rotation modifiers is not supported; preserving an original teacher through
those transformations needs additional handling.
Fine-grained tracing inside attention may require `attn_implementation="eager"`
when loading the model, so runtime attention masks match the traced implementation.

`LayerwiseQADModifier` remains an alias for old imports and recipe names, using
the new behavior. Remove `distill_teacher` from old `oneshot` calls.
