# AWQ Quantization #

Activation Aware Quantization (AWQ) is a state-of-the-art technique to quantize the weights of large language models which involves using a small calibration dataset to calibrate the model. The AWQ algorithm utilizes calibration data to derive scaling factors which reduce the dynamic range of weights while minimizing accuracy loss to the most salient weight values.

The AWQ implementation found in LLM Compressor is derived from the pioneering work of [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) and with assistance from its original maintainer, [@casper-hansen](https://github.com/casper-hansen).

## AWQ Recipe ##

AWQ is a **pre-quantization transform** — it computes and applies smoothing scales to model weights, but does not produce final quantized weights on its own. A downstream quantizer (`QuantizationModifier` or `GPTQModifier`) must follow AWQ in the recipe to finalize quantization.

```python
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]
```

The `scheme` on `AWQModifier` tells AWQ how the downstream quantizer will quantize, so that the grid search optimizes for the correct quantization format. It must match the downstream quantizer's scheme.

AWQ can also be stacked with other transforms and quantizers:
```python
recipe = [AWQModifier(...), GPTQModifier(...)]
recipe = [AWQModifier(...), QuantizationModifier(...)]
```

## Compressing Your Own Model ##
To use your own model, start with an existing example change the `model_id` to match your own model stub.
```python
model_id = "path/to/your/model"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
```

## Adding Mappings ##
In order to target weight and activation scaling locations within the model, the `AWQModifier` must be provided an AWQ mapping. For example, the AWQ mapping for the Llama family of models looks like this:

```python
[
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        "re:.*up_proj",
        ["re:.*down_proj"],
    ),
]
```

Note: the mappings define which layers get smoothed whereas targets and ignore define which layers get quantized. So if you include a layer in the ignore list that is going to get matched due to the included mappings, it will get smoothed but not quantized.

To support other model families, you can supply your own mappings via the `mappings` argument with instantiating the `AWQModifier`, or you can add them to the registry [here](/src/llmcompressor/modifiers/awq/mappings.py) (contributions are welcome!)
