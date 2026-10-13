# Norm distillation of compressed checkpoints

`NormDistillationModifier` improves a quantized model by training the gains of
its normalization layers (RMSNorm/LayerNorm `weight`), its weight quantization
scales, or both, so that its next-token distribution matches the unquantized
teacher's. Packed weights stay unchanged, so the result runs anywhere the
original checkpoint runs, with the same size and kernels.

The main use is a checkpoint that is already compressed, such as a published
W4A16 model. Load it with `from_pretrained` and pass the model object to
`oneshot`; it stays compressed in memory, decompresses on its first forward
pass and is compressed again on save:

```python
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.qad import NormDistillationModifier

model = AutoModelForCausalLM.from_pretrained(
    "RedHatAI/Qwen3-0.6B-quantized.w4a16", dtype="auto", device_map="auto"
)
oneshot(
    model=model,
    dataset=dataset,  # e.g. 1024-token WikiText-2 train blocks
    recipe=NormDistillationModifier(teacher="Qwen/Qwen3-0.6B"),
    num_calibration_samples=1760,
    max_seq_length=1024,
    batch_size=4,
    output_dir="Qwen3-0.6B-quantized.w4a16-norm-distilled",
)
```

[qwen3_example.py](qwen3_example.py) runs this end to end:

```bash
python qwen3_example.py --output ./qwen3-0.6b-w4a16-norm-distilled
```

The modifier can also follow a weight quantization modifier in the same recipe;
it trains after every other modifier has finalized.

Norm tuning is enabled by default. To tune only scales, use
`NormDistillationModifier(teacher=..., tune_norms=False, tune_scales=True)`.
Set both switches to `True` to tune norms and scales together. Scale tuning
updates `weight_scale` and, when present, `weight_global_scale` while retaining
the packed quantized values.

## How it works

- Targets: every module whose name ends in `norm` (`targets=["re:.*norm$"]`),
  for Qwen3 the input, post-attention, q/k and final norms. Only their 1-D
  `weight` is trained when `tune_norms=True`.
- Loss: token-averaged KL divergence from the teacher's next-token distribution
  to the student's, over the calibration batches, with Adam on FP32 master
  copies of the gains (`lr=1e-3`) and dynamic loss scaling for FP16 models.
- Selection: `validation_fraction=0.1` of the batches are held out, and the
  gains with the lowest validation KL after each epoch are kept. The original
  gains are a candidate, so a run that does not help leaves the model unchanged.
- Memory: the student and teacher run full forward and backward passes and must
  both fit on the execution device. The calibration batches are read from the
  dataloader; no calibration forward passes are needed.

Distill against the unquantized model's distribution, not the language-modeling
loss: tuning norms on a next-token loss mostly adapts the model to the
calibration domain, and the same tuning on the FP16 model gains more.

## Results

Qwen3-0.6B, 1760 WikiText-2 train blocks of 1024 tokens, batch 4, one epoch
(396 updates), teacher `Qwen/Qwen3-0.6B`. Perplexity in vLLM 0.10.2 (FP16) on
the WikiText-2 test set and 128 blocks of C4 validation, which is never used for
training. Two seeds each.

| Checkpoint | Norms | WikiText-2 PPL | C4 PPL |
|---|---|---:|---:|
| `JunHowie/Qwen3-0.6B-GPTQ-Int4` (GPTQ, group 128) | original | 29.17 | 35.60 |
| | distilled, seeds 0 / 1 | 25.55 / 25.41 (-12.4% / -12.9%) | 32.81 / 32.78 (-7.8% / -7.9%) |
| `RedHatAI/Qwen3-0.6B-quantized.w4a16` (GPTQ, group 64, asym) | original | 26.70 | 32.56 |
| | distilled, seeds 0 / 1 | 24.74 / 24.78 (-7.3% / -7.2%) | 31.45 / 31.51 (-3.4% / -3.2%) |

Training takes about 70 seconds per model on one H100.

The GPTQ-format checkpoint was first re-encoded as compressed-tensors W4A16 so
that llm-compressor can load it; every decompressed weight equals the GPTQ
dequantization bit for bit, and both versions score the same in vLLM.

Norm Tweaking (https://arxiv.org/abs/2309.02784) proposed tuning normalization
layers after quantization.
