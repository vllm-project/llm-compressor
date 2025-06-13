# Quantizing Models with Activation-Aware Quantization (AWQ) #

Activation Aware Quantization (AWQ) is a state-of-the-art technique to quantize the weights of large language models which involves using a small calibration dataset to calibrate the model. The AWQ algorithm utilizes calibration data to derive scaling factors which reduce the dynamic range of weights while minimizing accuracy loss to the most salient weight values.

The AWQ implementation found in LLM Compressor is derived from the pioneering work of [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) and with assistance from its original maintainer, [@casper-hansen](https://github.com/casper-hansen).

## AWQ Recipe ##

The AWQ recipe has been inferfaced as follows, where the `AWQModifier` adjusts model scales ahead of efficient weight quantization by the `QuantizationModifier`

```python
recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]
```

## Compressing Your Own Model ##
To use your own model, start with an existing example change the `model_id` to match your own model stub.
```python
model_id = "path/to/your/model"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
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

To support other model families, you can add supply your own mappings via the `mappings` argument with instantiating the `AWQModifier`, or you can add them to the registry [here](/src/llmcompressor/modifiers/awq/mappings.py) (contributions are welcome!)
