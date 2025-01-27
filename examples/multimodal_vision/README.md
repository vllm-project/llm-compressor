# Quantizing Multimodal Vision-Language Models #

<p align="center" style="text-align: center;">
    <img src=http://images.cocodataset.org/train2017/000000231895.jpg alt="sample image from MS COCO dataset"/>
</p>
<em>

``` 
<|system|>
You are a helpful assistant.

<|user|>
Please describe the animal in this image

<|assistant|>
The animal in the image is a white kitten.
It has a fluffy coat and is resting on a white keyboard.
The kitten appears to be comfortable and relaxed, possibly enjoying the warmth of the keyboard.
```
</em>

This directory contains example scripts for quantizing a variety of vision-language models using the GPTQ quantization. Most examples do not demonstrate quantizing separate vision encoder parameters if they exist, as compressing these parameters offers little benefit with respect to performance-accuracy tradeoff.

## Compressing Your Own Model ##
To use your own multimodal modal, start with an existing example change the `model_id` to match your own model stub.
```python3
model_id = "path/to/your/model"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
```

## Customizing GPTQModifier Parameters ##
The GPTQModifier is the modifier responsible for performing quantization of the model weights. For more information on quantizing with different weight schemes, see the `quantization_` examples in the [examples folder](/examples/).

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
Sequential targets are the modules which determine the granularity of error propagation and activation offloading when performing forward passes of the model. These are typically the "transformer blocks" of the model, also referred to as "layers" with llm-compressor.

Choosing sequential targets with higher granularity (for example "Linear" instead of "LlamaDecoderLayer") will result in fewer hessians being allocated at the same time, decreasing the memory requirements for compression. This may also increase the recovered accuracy of the model, as compression error is propagated at a higher granularity. However, using higher granularity sequential targets may also increase compression time, as more time is spent offloading and onloading activations.

### Ignore ###
If your model is not traceable for your desired dataset, first consider adding any problematic modules to the ignore list. Doing this prevents the model tracer from tracing the internals of those modules, thereby avoid the untraceable operations.

## Tracing Errors ##
Because the architectures of vision-language models is often times more complex than those of typical decoder-only text models, you may encounter `torch.fx.TraceError`s when attempting to quantize your model. For more information on `torch.fx.TraceError`s, why they occur, and how to resolve them, please see the [Model Tracing Guide](/src/llmcompressor/transformers/tracing/GUIDE.md).

## Adding Your Own Smoothquant Mappings ##
For a guide on adding smoothquant mappings for your dataset, see the [SmoothQuant Guide](/src/llmcompressor/modifiers/smoothquant/README.md).

## Adding Your Own Data Collator ##
Most examples utilize a generic `data_collator` which correctly correlates data for most multimodal datasets. If you find that your model needs custom data collation (as is the case with [pixtral](/examples/multimodal_vision/pixtral_example.py)), you can modify this function to reflect these model-specific requirements.