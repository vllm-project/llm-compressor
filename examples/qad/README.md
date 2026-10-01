# Block-wise quantization-aware distillation

`QADModifier` refines quantized weights by reconstructing decoder blocks'
unquantized outputs. Put it after the quantization method in the same recipe.
QAD shares that method's calibration data and uses the existing sequential
pipeline. It does not load a separate teacher model.

`sequential_targets_per_subgraph` sets how many consecutive decoder blocks QAD
trains jointly. Each block takes the previous block's fake-quantized output, and
the loss compares the last block's output with its unquantized output. Training
memory grows with the block count, since backpropagation runs through every
block in the subgraph.

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
            lr=2e-6,
            gradient_accumulation_steps=4,
        ),
    ],
    pipeline="sequential",
    sequential_targets=["LlamaDecoderLayer"],
    sequential_targets_per_subgraph=1,  # decoder blocks trained jointly
    propagate_error=True,  # default; recommended for QAD
    num_calibration_samples=512,
    max_seq_length=2048,
    batch_size=1,
)
```

See [llama3_example.py](llama3_example.py) for a complete NVFP4A16 example with
BF16 model execution and activations:

```bash
python llama3_example.py --quantizer rtn --output ./llama-rtn-qad
python llama3_example.py --quantizer gptq --output ./llama-gptq-qad
python llama3_example.py --quantizer gptq --targets-per-subgraph 2 --output ./llama-gptq-qad-2
```

The example defaults to one UltraChat dataset for calibration and QAD. Override it with
`--dataset`, `--split`, and `--text-column`; `--samples` and `--max-seq-length`
apply to both stages. Chat datasets use the model's chat template.

## ModelOpt data mix

Use `--dataset modelopt_qad_mix` to load the seven-source blend from
[NVIDIA Model Optimizer's QAT/QAD example](https://github.com/NVIDIA/Model-Optimizer/blob/6e4789fa43726f800b6d6f63d6611b6472b00ba0/examples/llm_qat/configs/dataset/blend.yaml).
The loader lives in [modelopt_mix.py](modelopt_mix.py); both the preceding
quantizer and QAD receive the resulting dataset.

| Hugging Face dataset | Split | Relative weight | Samples when requesting 1024 |
| --- | --- | ---: | ---: |
| `nvidia/Nemotron-SWE-v1` | `r2e_gym` | 6000 | 323 |
| `nvidia/Nemotron-Math-v2` | `medium` | 2500 | 135 |
| `nvidia/Nemotron-Science-v1` | `MCQ` | 1500 | 81 |
| `nvidia/Nemotron-Science-v1` | `RQA` | 1500 | 81 |
| `nvidia/Nemotron-Instruction-Following-Chat-v1` | `chat_if` | 5000 | 269 |
| `nvidia/Nemotron-Post-Training-Dataset-v2` | `chat` | 1500 | 81 |
| `nvidia/Nemotron-Competitive-Programming-v1` | `competitive_coding_python_part00` | 1000 | 54 |

Run from this directory, selecting the sample count and epoch budget:

```bash
python llama3_example.py --dataset modelopt_qad_mix --quantizer gptq --samples 512 --epochs 3 --max-seq-length 2048 --output ./llama-mix-512-e3
python llama3_example.py --dataset modelopt_qad_mix --quantizer gptq --samples 1024 --epochs 7 --max-seq-length 2048 --output ./llama-mix-1024-e7
```

Weights specify proportions of conversations, not tokens. The loader rounds
allocations to exactly `--samples`, streams pinned dataset revisions with seed
42 and a bounded shuffle buffer, applies the model's chat template, and truncates
to `--max-seq-length`. It records source revisions and counts in
`dataset.info.description`. `--split` and `--text-column` apply only to a single
Hugging Face dataset, not this preset. Access to
`nvidia/Nemotron-Post-Training-Dataset-v2` requires an authorized Hugging Face
token; loading fails if any required source is inaccessible or too short.

This example adopts the upstream sources and mixture weights. QAD uses its own
validation split described below; `--epochs` is the per-subgraph epoch count, with
the best validation weights restored before propagation. These
commands demonstrate loading and training, not the full configuration used for
the long-context epoch-ablation benchmark results.

## Execution

For each sequential subgraph of one or more consecutive decoder blocks:

1. During calibration, hooks cache the blocks' inputs and the last block's
   unquantized outputs. The preceding method collects its statistics in the same
   forward pass.
2. At the existing `sequential_epoch_end` event, preceding modifiers finish
   preparing the blocks' quantized weights and quantization parameters.
3. QAD replays the blocks as a chain with fake quantization, feeding each block
   the previous block's output, and minimizes the last block's output MSE.
   Only weights configured for quantization are trained. Calibration and capture
   hooks are disabled throughout QAD training and validation. Weight observers
   update qparams before training and after each epoch.
4. QAD restores the weights with the lowest validation loss, which may be the
   untrained weights, re-observes their qparams, materializes quantized weights,
   and releases the training cache. The pipeline propagates the final outputs to
   the next subgraph.

The target is **local**: the original current blocks receive the same cached
input as the student. With subgraph function `f` (the composition of its
blocks), original weights `W0`, student weights `W`, and cached input `h`, QAD
fits `f(h, Q(W))` to `f(h, W0)`. Inside the student chain, later blocks take the
earlier blocks' quantized outputs. This captures the current blocks before the
preceding quantizer changes them; it does not reconstruct an independent
original-model activation stream.

`propagate_error` selects `h`. With the default `propagate_error=True`
(recommended), `h` is the preceding subgraphs' quantized output, so each subgraph
trains on the inputs it receives at inference. With `propagate_error=False`,
`h` is the original model's activation, which gives plain block-wise
reconstruction.

Replay uses each complete target module's `forward`, retaining its internal
attention, MLP, branches, and residual connections. Multiple target modules in a
subgraph train jointly only as a chain, in which each module's first input is
the previous module's output, as with decoder layers. QAD rejects other subgraph
structures.

## Composing quantization methods

QAD does not depend on GPTQ Hessians or quantizer class names. Tiny Llama
integration tests cover RTN (`QuantizationModifier`), GPTQ (`GPTQModifier`), and
AWQ followed by RTN, each followed by QAD, with NVFP4A16 and integer W4A16.
Another method can precede QAD if it:

- Attaches a compressed-tensors weight quantization scheme during initialization.
- Preserves the unquantized block computation during the calibration pass used
  to capture teacher targets.
- Prepares its weights and valid qparams in `sequential_epoch_end`, before QAD's
  callback runs. Place QAD after all modifiers that prepare the block.
- Leaves floating-point weights compatible with compressed-tensors fake
  quantization, and suppresses calibration hooks during its own replay forwards.

A transform such as AWQ or SmoothQuant must be composed with a weight
quantization modifier. For example, the tested AWQ recipe is:

```python
from llmcompressor.modifiers.transform.awq import AWQModifier

recipe = [
    AWQModifier(n_grid=4),
    QuantizationModifier(scheme="NVFP4A16", ignore=["lm_head"]),
    QADModifier(),
]
```

Pass this recipe to `oneshot` with the sequential settings above. The AWQ
integration tests check per-block optimization, cache cleanup, and finite model
outputs; save/reload/generation tests separately cover RTN and GPTQ. Other
transform/quantizer combinations need their own validation. Methods that change
target boundaries or overwrite teacher weights before capture do not satisfy the
contract automatically.

## Training and supported scope

- Use single-process sequential calibration and one complete target module per
  subgraph. `propagate_error=True` (the default) is recommended; see
  [Execution](#execution). Quantized weights outside the selected targets, such
  as a quantized LM head, keep the preceding modifier's result. Repeated calls
  to the same target in a calibration batch are unsupported.
- The same samples, preprocessing, sequence lengths, and loader batch size feed
  both calibration and QAD. There is no `qad_dataset` or `teacher_mode` option.
  `gradient_accumulation_steps` controls the QAD update batch size.
- At least two calibration batches are required. A deterministic split with
  `validation_fraction=0.1` holds batches out from QAD gradient updates, although
  they still participate in quantizer calibration. Keep downstream test data
  separate.
- Each block has its own optimizer and trains for `num_epochs` epochs. The
  weights with the lowest validation loss are then restored; the quantizer's
  initial weights are included among the candidates.
- FP16/BF16 execution uses FP32 optimizer master weights. FP16 backward also
  uses dynamic loss scaling to retain small reconstruction gradients. Overflow
  retries use the same accumulation group with a lower scale, up to 32 attempts.
  Gradients are unscaled before clipping with `max_grad_norm=1.0` by default;
  set it to `None` to disable clipping.
- Inputs, targets, and best-weight snapshots default to CPU storage through
  `offload_device="cpu"`. QAD still needs memory for a block's backward
  pass and optimizer state.
- QAD stores independent input and teacher snapshots in the shared
  `IntermediatesCache`. Set `sequential_prefetch=True` to prefetch QAD batches
  while preserving the training shuffle and validation split.
- QAD computes reconstruction MSE over all output positions.
- Weight re-observation follows [Charles's schedule](https://github.com/vllm-project/llm-compressor/pull/3051): before training,
  after every epoch, and before final materialization. Set
  `reobserve_weights=False` for fixed weight qparams. Scales are not optimized
  by gradient descent. QAD does not re-observe activations; current validation
  covers weight-only quantization. Activation quantization needs separate
  gradient and calibration validation.
- Re-observation requires the preceding method to retain live weight observers.
  All modules are observed before any qparams are updated, preserving shared
  global scales in fused NVFP4 projections. Best-checkpoint restoration restores
  weights only; final re-observation then recomputes qparams from them and can
  change validation MSE again, so the final loss is logged separately from the
  best loss. Per-target updates, validation losses, and re-observation stages are
  reported through the log.

Save the resulting model through the normal compressed checkpoint path.
