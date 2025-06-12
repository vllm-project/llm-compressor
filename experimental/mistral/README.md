# Mistral-format model compression (experimental)

This folder contains tools for compressing Mistral-format models, like `mistralai/Devstral-Small-2505` and `mistralai/Magistral-Small-2506`.

## FP8 W8A8 Quantization

This script quantizes Mistral-format models to FP8. It is not for use with HuggingFace-format models.

### 1. Download the model

Download the model and save it to a new "FP8" folder. We use `mistralai/Magistral-Small-2506` as an example.

```bash
huggingface-cli download mistralai/Magistral-Small-2506 --local-dir Magistral-Small-2506-FP8
```

### 2. Clean up HuggingFace-specific files

Models from the Hub often include files for both the native Mistral format and the HuggingFace `transformers` format. This script works on the native format, so the `transformers` files should be removed to avoid confusion.

The HuggingFace-specific files are typically `config.json`, `model-000*-of-000*.safetensors`, and `model.safetensors.index.json`. The `params.json`, `tekken.json` and `consolidated.safetensors` are for the native format.

Before deleting, it's a good idea to look at the files in the directory to understand what you're removing.

Once you're ready, remove the `transformers`-specific files:

```bash
rm Magistral-Small-2506/config.json Magistral-Small-2506/model.safetensors.index.json Magistral-Small-2506-FP8/model-000*
```

### 3. Run the quantization script

Now, run the FP8 quantization script on the directory. This will modify the `.safetensors` files in-place and update `params.json` and `consolidated.safetensors`.

```bash
python fp8_quantize.py Magistral-Small-2506-FP8
```

### 4. Use the quantized model

The model should now be ready to use in vLLM!

```bash
vllm serve Magistral-Small-2506-FP8 --tokenizer-mode mistral --config-format mistral --load-format mistral --tool-call-parser mistral --enable-auto-tool-choice
```