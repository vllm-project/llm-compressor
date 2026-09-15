# Block-wise quantization-aware distillation

`QADModifier` refines quantized weights by reconstructing each decoder block's
unquantized outputs. Put it after the quantization method in the same recipe.
QAD shares that method's calibration data and uses the existing sequential
pipeline. It does not load a separate teacher model.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Choose a preceding weight quantization method.
quantizer = QuantizationModifier(scheme="NVFP4A16", ignore=["lm_head"])  # RTN
# quantizer = GPTQModifier(scheme="NVFP4A16", ignore=["lm_head"], actorder="static")

oneshot(
    model=model,
    dataset=dataset,
    recipe=[
        quantizer,
        QADModifier(
            num_epochs=3,
            learning_rate=2e-6,
            gradient_accumulation_steps=4,
        ),
    ],
    pipeline="sequential",
    sequential_targets=["LlamaDecoderLayer"],
    sequential_targets_per_subgraph=1,
    propagate_error=True,
    num_calibration_samples=512,
    max_seq_length=2048,
    batch_size=1,
)
```

See [llama3_example.py](llama3_example.py) for a complete NVFP4A16 example:

```bash
python llama3_example.py --quantizer rtn --output ./llama-rtn-qad
python llama3_example.py --quantizer gptq --output ./llama-gptq-qad
```

The example uses one UltraChat dataset for calibration and QAD. Override it with
`--dataset`, `--split`, and `--text-column`; `--samples` and `--max-seq-length`
apply to both stages. Chat datasets use the model's chat template.

## Execution

For each decoder block:

1. During calibration, hooks cache the block's inputs and unquantized outputs.
   The preceding method collects its statistics in the same forward pass.
2. At the existing `sequential_epoch_end` event, preceding modifiers finish
   preparing the block's quantized weights and quantization parameters.
3. QAD replays the block with fake quantization and minimizes masked output MSE.
   Only weights configured for quantization are trained. Calibration and capture
   hooks are disabled throughout QAD training and validation. Weight observers
   update qparams before training and after each epoch.
4. QAD restores the best validation weights and their matching qparams,
   re-observes weights, materializes quantized weights, and releases the training cache. The
   pipeline propagates the final outputs to the next block.

The target is **local**: the original current block receives the already
quantized upstream outputs. With block function `f`, original weights `W0`,
student weights `W`, and cached input `h`, QAD fits `f(h, Q(W))` to `f(h, W0)`.
This captures the current block before the preceding quantizer changes it;
it does not reconstruct an independent original-model activation stream.

Replay uses the complete target module's `forward`, retaining its internal
attention, MLP, branches, and residual connections. Multiple target modules
cannot be jointly optimized as an arbitrary traced subgraph in this version.

## Composing quantization methods

QAD does not depend on GPTQ Hessians or quantizer class names. RTN
(`QuantizationModifier`) and GPTQ (`GPTQModifier`) are tested with NVFP4A16 and
integer W4A16. Another method can precede QAD if it:

- Attaches a compressed-tensors weight quantization scheme during initialization.
- Preserves the unquantized block computation during the calibration pass used
  to capture teacher targets.
- Prepares its weights and valid qparams in `sequential_epoch_end`, before QAD's
  callback runs. Place QAD after all modifiers that prepare the block.
- Leaves floating-point weights compatible with compressed-tensors fake
  quantization, and suppresses calibration hooks during its own replay forwards.

A transform such as AWQ or SmoothQuant is not a weight quantizer by itself; it
must be composed with a quantization modifier. Those combinations require their
own validation before being claimed as supported. Methods that change target
boundaries, overwrite teacher weights before capture, or disable sequential
error propagation do not satisfy the contract automatically.

## Training and supported scope

- Use single-process sequential calibration, one complete target module per
  subgraph, and `propagate_error=True`. All quantized weights must belong to the
  selected targets. Ignore the output LM head. Cross-block weight sharing and
  repeated calls to the same target in a calibration batch are unsupported.
- The same samples, preprocessing, sequence lengths, and loader batch size feed
  both calibration and QAD. There is no `qad_dataset` or `teacher_mode` option.
  `gradient_accumulation_steps` controls the QAD update batch size.
- At least two calibration batches are required. A deterministic split with
  `validation_fraction=0.1` holds batches out from QAD gradient updates, although
  they still participate in quantizer calibration. Keep downstream test data
  separate.
- Each block has its own optimizer, `num_epochs`, and early stopping
  (`early_stopping_patience=3`, `validation_relative_min_delta=0.001`). The
  quantizer's initial weights are included among the best-weight candidates.
- FP16/BF16 execution uses FP32 optimizer master weights. Gradients are clipped
  with `max_grad_norm=1.0` by default; set it to `None` to disable clipping.
- Inputs, targets, and best-weight snapshots default to CPU storage through
  `target_offload_device="cpu"`. QAD still needs memory for a block's backward
  pass and optimizer state.
- With `use_loss_mask=True`, QAD reuses the pipeline's per-batch `loss_mask`.
  Without it, all output positions contribute to MSE.
- Weight re-observation follows [Charles's schedule](https://github.com/vllm-project/llm-compressor/pull/3051): before training,
  after every epoch, and before final materialization. Set
  `reobserve_weights=False` for fixed weight qparams. Scales are not optimized
  by gradient descent. QAD does not re-observe activations; current validation
  covers weight-only quantization. Activation quantization needs separate
  gradient and calibration validation.
- Re-observation requires the preceding method to retain live weight observers.
  All modules are observed before any qparams are updated, preserving shared
  global scales in fused NVFP4 projections. Best-checkpoint restoration includes
  weight scales, zero points, and global scales. Final re-observation can change
  validation MSE again; the final loss is logged separately from the best loss.

`optimizer_steps`, `epochs_completed`, `best_validation_losses`, and
`validation_histories`, and `reobservations` are keyed by target module name. Save the resulting model
through the normal compressed checkpoint path. `LayerwiseQADModifier` remains
an alias for `QADModifier`.
