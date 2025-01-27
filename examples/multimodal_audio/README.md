# Quantizing Multimodal Audio Models #

<audio controls>
    <source src="https://datasets-server.huggingface.co/cached-assets/MLCommons/peoples_speech/--/f10597c5d3d3a63f8b6827701297c3afdf178272/--/clean/test/0/audio/audio.wav?Expires=1738010344&Signature=V6eMq7mQo1~wrkdswghsWaf9aklEQwoqw8FwJUiHAL75K7BcarTepBYcQkFIRi6usgU5J0TlX~wBwIlobAE7GzEXTUI7j5KA1MbFTiLo-nIYiq-WpA70EHW3mGy5HyCm01wKD49ngQDOgHX0-NrvTuXJCkTBhfYBwbQ5QsM8Wv3sbgEyadE~RMEGJLTfQL5fzQp3l1FWMdGuBJHDqSZa1SzTbOJYfmNQjGlfgWpm8Fhf5KWDl1NQSgWaiWRC0evbxt~C9Z8sEYwIEma7tTJafWqc2T9Awn8RdMqNKXnqSZ-mQBBxWVAV9cJbGKsj5JXJJwMPl23AUpzfSale71602g__&Key-Pair-Id=K3EI6M078Z3AC3">
    Your browser does not support the audio element.
</audio>
<em>

``` 
<|startoftranscript|> <|en|>
...

<|transcribe|> <|notimestamps|>
that's where you have a lot of windows in the south no actually that's passive solar
and passive solar is something that was developed and designed in the 1960s and 70s
and it was a great thing for what it was at the time but it's not a passive house
```
</em>

This directory contains example scripts for quantizing a variety of audio language models using the GPTQ quantization.

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
        sequential_targets=["WhisperEncoderLayer", "WhisperDecoderLayer"],
        ignore=["lm_head"],
    )
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