# Quantizing Multimodal Vision-Language Models #
This directory contains example scripts for quantizing a variety of vision-language models using the GPTQ W4A16 quantization scheme.

## Using your own models ##

```python3
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        sequential_targets=["MistralDecoderLayer"],
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
    ),
]
```

### Sequential Targets ###

### Ignore ###

### Tracing Errors ###
Because the architectures of vision-language models is often times more complex than those of typical decoder-only text models, you may encounter `torch.fx.TraceError`s when attempting to quantize your model. For more information on `torch.fx.TraceError`s, why they occur, and how to resolve them, please see the [Model Tracing Guide](/src/llmcompressor/transformers/tracing/README.md).

### Adding Smoothquant Mappings ###

### Adding Data Collator ###
* TODO: create a default "multimodal" collator

## Customizing Dataset and Quantization Scheme ##
. For a detailed walkthrough of customzing datasets and quantization for W4A16, see the
[Quantization Guide](/examples/quantization_w4a16/README.md).