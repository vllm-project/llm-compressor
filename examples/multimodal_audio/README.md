# Quantizing Multimodal Audio Models #

https://github.com/user-attachments/assets/6732c60b-1ebe-4bed-b409-c16c4415dff5

Audio provided by Daniel Galvez et al. under creative commons license

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
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
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

## Adding Your Own Smoothquant Mappings ##
For a guide on adding smoothquant mappings for your dataset, see the [SmoothQuant Guide](/src/llmcompressor/modifiers/smoothquant/README.md).

## Adding Your Own Data Collator ##
Most examples utilize a generic `data_collator` which correctly correlates data for most multimodal datasets. If you find that your model needs custom data collation (as is the case with [pixtral](/examples/multimodal_vision/pixtral_example.py)), you can modify this function to reflect these model-specific requirements.

## Sample Audio Provided Under a Creative Commons Attribution License ##
https://creativecommons.org/licenses/by/4.0/legalcode
```
@article{DBLP:journals/corr/abs-2111-09344,
  author    = {Daniel Galvez and
               Greg Diamos and
               Juan Ciro and
               Juan Felipe Cer{\'{o}}n and
               Keith Achorn and
               Anjali Gopi and
               David Kanter and
               Maximilian Lam and
               Mark Mazumder and
               Vijay Janapa Reddi},
  title     = {The People's Speech: {A} Large-Scale Diverse English Speech Recognition
               Dataset for Commercial Usage},
  journal   = {CoRR},
  volume    = {abs/2111.09344},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.09344},
  eprinttype = {arXiv},
  eprint    = {2111.09344},
  timestamp = {Mon, 22 Nov 2021 16:44:07 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-09344.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```