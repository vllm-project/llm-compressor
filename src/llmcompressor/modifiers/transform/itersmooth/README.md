# IterSmooth

`IterSmoothModifier` performs calibration-based outlier suppression before weight
and activation quantization. It supports fused norm-linear, linear-linear, gated
up-down, and value-output attention mappings, including GQA/MQA.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model=model,
    processor=tokenizer,
    dataset=dataset,
    recipe=[
        IterSmoothModifier(alpha=0.9),
        QuantizationModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
    ],
    pipeline="independent",
)
```

Use independent calibration epochs so the quantizer observes smoothed inputs.
This implementation computes analytic scales once per mapped subgraph rather
than running an iterative optimizer. Bias-free RMSNorm models require the default
`symmetric=True`; non-fused runtime hooks are not supported.

See [the Qwen3.6-27B example](../../../../../examples/quantization_w8a8_int8/qwen3_6_itersmooth_example.py)
for model loading, calibration data preparation, and the complete quantization recipe.
