# Layerwise MSE QAD

`LayerwiseQADModifier` performs block-local quantization-aware distillation.
Each decoder block is initialized by GPTQ, then optimized against the matching
full-precision teacher block using hidden-state mean squared error.

```python
from llmcompressor import oneshot

recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      scheme: W4A16
      ignore: [lm_head]
    LayerwiseQADModifier:
      num_epochs: 12
      learning_rate: 2.0e-6
      gradient_accumulation_steps: 4
      max_grad_norm: 1.0
      validation_fraction: 0.1
      early_stopping_patience: 3
      validation_relative_min_delta: 0.001
"""

model = oneshot(
    model="student-model-or-path",
    distill_teacher="full-precision-teacher-model-or-path",
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=recipe,
    pipeline="sequential",
    propagate_error=True,
    sequential_targets_per_subgraph=1,
    batch_size=2,
    num_calibration_samples=128,
    max_seq_length=2048,
)
```

The calibration dataloader `batch_size` is the QAD microbatch size. One
optimizer step combines `gradient_accumulation_steps` microbatches. Every block
is visited once. Cached calibration batches are deterministically split into
90% training and 10% validation data. Validation MSE is evaluated after every
epoch. Any lower validation MSE updates the saved best weights, while patience
is reset only after a cumulative relative improvement of
`validation_relative_min_delta`. This tolerates short-term fluctuations and
avoids extending training for numerical noise. The weights with the lowest
observed validation MSE are restored before quantization.

Teacher and student receive the same block input. After the current block is
optimized, its quantized output is propagated to the next block. This method
uses hidden-state MSE, not final-logits KL, and therefore is not equivalent to
end-to-end QAD.
